from os import listdir
from os.path import isfile, join
from to_sql import to_sql

current_dir 	= os.path.dirname(__file__)

path 		= os.path.abspath(os.path.join(current_dir, "results"))

results_files 	= [files for files in listdir(path) if isfile(join(path, files))]


for results in results_files
	result_dict = json.load(results)
	to_sql(program_results)
