from functions import *
import os

start_time = time.time()

# ......................................................
#
#   PARSE ARGUMENTS ----
#
# ......................................................
parser = argparse.ArgumentParser(description='Perform univariate statistical test such as T test or Zero inflated Wilcoxon.')
parser.add_argument('-i', '--input', type=str, help='Input count matrix path.')
parser.add_argument('-o', '--output_folder', type=str, help='Output folder path.')
parser.add_argument('-t', '--top_tags', type=float, default=200000, help='Top tags with best test statistics to keep.')
parser.add_argument('-c', '--chunk_size', type=int, default=10000, help='Size of each chunk in number of rows.')
parser.add_argument('-p', '--processes', type=int, default=mp.cpu_count(), help='Number of CPUs used/Default: number of CPUs available')
parser.add_argument('-d', '--condition_folder', type=str, help='Path to the condition folder.')
parser.add_argument('-m', '--cpm', nargs='?', const='default', type=str, help='Perform CPM normalization with optional file argument')
parser.add_argument('--test_type', choices=['ttest', 'pitest', 'ziw','wilcoxon','variance'], default='ttest', help='Test to perform and rank results.')
args = parser.parse_args()

# logging.basicConfig(filename = "chunk_processing.log", level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

# Get the directory where the script is executed
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the log file path in the same directory as the script
log_file_path = os.path.join(script_directory, "chunk_processing.log")

# Configure logging to overwrite the log file if it already exists
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Use 'w' to overwrite the existing log file
)

logging.info("Log file created and ready for new logging.")

# ......................................................
#
#   GUESS INPUT SEPARATOR ----
#
# ......................................................
with open(args.input) as f:
    head_lines = [next(f).rstrip() for x in range(2)]

common_delimiters = [',', ';', '\t', ' ', '|', ':']
for d in common_delimiters:
    ref = head_lines[0].count(d)
    if ref > 0:
        if all([ ref == head_lines[i].count(d) for i in range(1, 2)]):
            input_separator = d

# ......................................................
#
#   ESTIMATE TOTAL NUMBER OF TAGS (LINES) ----
#
# ......................................................

# Estimate the total number of lines (replacing the slower previous method)
total_tags = estimate_total_lines(args.input)

# Calculate the number of top tags at the start
if 0 < args.top_tags <= 1:
    # Convert percentage to an absolute number at the start
    args.top_tags = int(args.top_tags * total_tags)
else:
    args.top_tags = int(args.top_tags)
# ......................................................
#
#   EXECUTE STAT TEST ----
#
# ......................................................
with open(args.input, 'r') as file:
    header = file.readline().strip().split(input_separator)

# Create a pool of processes
pool = mp.Pool(processes = args.processes)

top_tags_list = []
condition_files = glob.glob(os.path.join(args.condition_folder, "*.tsv"))
# For each condition, the matrix is sliced into chunks and each chunk is processed individually in a process
condition_files = [file for file in condition_files if 'train' in os.path.basename(file)]
for condition_file in condition_files:
    # Create the data dictionary from the condition file
    data_dict = create_data_dict(condition_file, args.test_type)

    # Prepare the worker function which will be executed in a process
    func = functools.partial(work_for_parallel_processes, data_dict, cpm_normalization = args.cpm, header = header, test_type = args.test_type)

    # result contains the stat test and the tag of all chunks
    result = pool.imap(func, pd.read_csv(args.input, sep = input_separator, chunksize = args.chunk_size))

    # For each chunk result (stored into result object), only the top k-mers are selected
    top_tags = []
    if args.test_type == 'pitest':
        for chunk_results in result:
            top_tags.extend([(tag_values, pivalue, log2fold_change) for tag_values, pivalue, log2fold_change in chunk_results])

            top_tags.sort(key = lambda x: x[1])  # Sort based on pi_value
            top_tags = top_tags[:args.top_tags]
        top_tags_list.append(top_tags)

    elif args.test_type == 'ttest' or args.test_type == 'wilcoxon':
        for chunk_results in result:
            top_tags.extend([(t_statistic, tag_values) for t_statistic, tag_values in chunk_results])
            top_tags.sort(key = lambda x: x[0], reverse = True)  # Sort based on t-statistic
            top_tags = top_tags[:args.top_tags]
        top_tags_list.append(top_tags)

    elif args.test_type == 'ziw':
        for chunk_results in result:
            top_tags.extend([(test_statistic, tag_values) for test_statistic, tag_values in chunk_results])

            top_tags.sort(key = lambda x: x[0], reverse = True)  # Sort based on test-statistic
            top_tags = top_tags[:args.top_tags]
        top_tags_list.append(top_tags)

    elif args.test_type == 'variance':
        for chunk_results in result:
            print(chunk_results)
            top_tags.extend([(variance_value, tag_values) for variance_value, tag_values in chunk_results])
            top_tags.sort(key=lambda x: x[0], reverse=True)  # Sort based on variance (descending order)
            top_tags = top_tags[:args.top_tags]
        top_tags_list.append(top_tags)

    logging.info("Condition file treated")

if args.test_type == 'ttest' or args.test_type == 'wilcoxon' or args.test_type == 'ziw':
    header.append('test_statistic')

elif args.test_type =='pitest':
    header.append('pivalue')
    header.append('log2foldchange')
elif args.test_type == 'variance':
    header.append('variance')

# ......................................................
#
#   OUTPUT TOP K-MERS ----
#
# ......................................................
# Save the top tags for each condition file to separate output files
try:
    # Delete output folder if it exists
    shutil.rmtree(args.output_folder)
except OSError:
    pass

os.makedirs(args.output_folder)

for condition_file, top_tags in zip(condition_files, top_tags_list):
    condition_name = os.path.basename(condition_file)
    condition_name = os.path.splitext(condition_name)[0]  # Remove file extension if present

    output_data = []
    output_file = os.path.join(args.output_folder, f"{condition_name}.txt")
    with open(output_file, 'w') as file:
        file.write(' '.join(header) + '\n')

        if args.test_type == 'ttest' or args.test_type == 'wilcoxon':
            for t_statistic, values in top_tags:
                output_data.append([values, t_statistic])
                file.write(' '.join(str(x) for x in output_data[-1]) + '\n')  # to include the tstat in the output table

        elif args.test_type == 'ziw':
            for test_statistic, values in top_tags:
                output_data.append([values, test_statistic])
                file.write(' '.join(str(x) for x in output_data[-1]) + '\n')  # to include the tstat in the output table

        elif args.test_type == 'pitest':
            for values, pivalue,log2fold_change in top_tags:
                output_data.append([values, pivalue, log2fold_change])
                file.write(' '.join(str(x) for x in output_data[-1]) + '\n')

        elif args.test_type == 'variance':
            for variance_value, values in top_tags:
                output_data.append([values, variance_value])
                file.write(' '.join(str(x) for x in output_data[-1]) + '\n')  # to include the variance in the output table

end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {:.3f} s".format(execution_time))
