from learning import model_generator
import argparse
import os, logging, time, json, time
from fetch_data import subject_generator

USERNAME = os.environ['USER']

def main(parsed):
	
	if parsed['unlabel_weight']:
		unlabel_weight = map( lambda x: float(x)/10, range(0,11)) + range(1,11)
	else:
		unlabel_weight = [1]
	# delete an element of a dictionary mutates the dictionary such that anybody which references this dictionary will also be affected (is not an issue for us)
	program_results = {'subjects' : {}, 'unlabel_weight': parsed['unlabel_weight']}
	del parsed['unlabel_weight']
	
	parameters 	= {"n_components_": range(4,41), "unlabel_weight": unlabel_weight}
	
	
	
	time_start = time.time()
	
	# as subject_generator is a generator, the program_results will be updated as specified how to update a dictionary using an iterator
	program_results['subjects'].update(**{subject_name: model_generator(parsed, parameters, d_val, d_test, subject_name) for d_val, d_test, subject_name in subject_generator(parsed['test_run']) })
	dt 		= time.time() - time_start
	
	# time.gmtime converts seconds to correct format for strftime.
	dt 		= time.strftime("%m-%d_%H-%M-%S", time.gmtime(dt))
	
	logging.info("Finished treating all subjects. Total run time: {0}".format(dt))
	program_results['program_duration']	= dt
	program_results['n_split']		= parsed['n_split']
	program_results['cov_type']		= parsed['cov_type']
	program_results['label_type']		= parsed['label_type']
	program_results['unlabel_independent']	= parsed['unlabel_independent']
	program_results['prior_loss_on']	= parsed['prior_loss_on']
	
	
	timer			= time.time()
	# time.gmtime converts seconds to correct format for strftime.
	timer 			= time.strftime("%m-%d-%H-%M-%S", time.gmtime(timer))
	
	# using __file__ provides current directory
	current_dir 		= os.path.dirname(__file__)
	if 'SYNTH_DATA' in program_results['subjects'].keys():
		dict_file_name 		= os.path.abspath(os.path.join(current_dir, "results", str(USERNAME) + "_" + "SYNTH_DATA.json"))
	else:
		dict_file_name 		= os.path.abspath(os.path.join(current_dir, "results", str(USERNAME) + "_" + timer + ".json"))
	
	with open(dict_file_name, 'w') as outfile:
			json.dump(program_results, outfile)

if __name__ == "__main__":
	"""
	Parser handles arguments passed from the terminal
	"""
	# ArgumentDefaultsHelpFormatter adds default info on calling help
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--n_split", type = int,  help = "Integer specifying number of splits used for VALIDATION", default = 5)
	
	parser.add_argument("--cov_type", type = str, help = "Specify enforced covariance type of GMM models" , default = "diag_tied", choices = ["full", "full_tied", "diag", "diag_tied", "spherical", "spherical_tied"])
	
	parser.add_argument("--n_init", type = int, help = "Integer specifying number of initialization of the model" , default = 1)
	
	parser.add_argument("--label_type", type = str, help = "Specify which kind of labels to use" , default = "y_c4", choices = ["y_c4","y_c6"])
	
	parser.add_argument("--test_run", action = 'store_true', help = "Specify to run the program using only one subject (For testing)")
	
	parser.add_argument("--max_iter", type = int, default = int(1e3), help = "Integer specifying number of maximum iterations of the model")
	
	parser.add_argument("--tolerance", type = float, default = 1e-6, help = "Float specifying tolerance of the model")
	
	parser.add_argument("--unlabel_independent", action = 'store_true', help = "Specify to run the program where unlabels are considered to have class latent variables or not (whether or not they are allowed to contribute to p(y|z) updates)")
	
	parser.add_argument("--unlabel_weight", action = 'store_true', help = "Specify whether we should weight the importance of including unlabeled data - done as a parameter search. \n\t - Range: [0,1]")
	
	parser.add_argument("--enable_progress_bar", action = 'store_true', help = "Specify whether we show a progress bar - DO NOT SET IF RUNNING ON HPC")

	parser.add_argument("--reg_term", type = float, default=1e-10, help = "Float specifying a regularizer term added to the diagonal of covariances matrices")
	
	parser.add_argument("--prior_loss_on", action = 'store_true', help = "Specify whether to use a loss function when predicting.")
	
	args 		= parser.parse_args()
	arg_dict	= {arg: getattr(args, arg) for arg in vars(args)}
	logging.basicConfig(format = '%(asctime)s, %(levelname)s, %(name)s, -- %(message)s',
			    datefmt = "%m-%d %H:%M:%S",
			    level = logging.DEBUG)
			    
	logging.info("Beginning semi-supervised learning with following user specified settings:\n\t - n_split: {n_split} \n\t - cov_type: {cov_type} \n\t - n_init: {n_init} \n\t - label_type: {label_type} \n\t - Test run: {test_run} \n\t - Maximum iterations: {max_iter} \n\t - Tolerance: {tolerance} \n\t - unlabel_weight: {unlabel_weight} \n\t - enable_progress_bar: {enable_progress_bar} \n\t - unlabel_independent: {unlabel_independent} \n\t - Use prior loss: {prior_loss_on}".format(**arg_dict) )

	if arg_dict['enable_progress_bar'] == False:
		logging.getLogger().setLevel(logging.CRITICAL)
	
	main(arg_dict)
