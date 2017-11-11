# joblib allows easy to use CPU parallel computing
# For far more control of the process, use multiprocessing

from joblib import Parallel, delayed
from sklearn.decomposition import NMF
import scipy.io as sio
import numpy as np
import json, os
import pandas as pd
import logging
import time
import argparse
import linecache, sys


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    logging.error('EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj))

# using __file__ provides current directory
current_dir 		= os.path.dirname(__file__)
#os.path.join uses the appropriate system child folder specification (i.e. "/,\,..")
mat_file_name 		= os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, "Data", "MatLabFiles", "SpecsAndLabels.mat"))

nmf_file_name 		= [os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, "Data","SubjectData", "NMFandLabelsSubject_" + str(sub) + ".gzip" )) for sub in range(19)]

reconstruct_file_name 	= [os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, "Data","SubjectData","ReconstructedDataSubject_" + str(sub) + ".gzip")) for sub in range(19)]

dict_file_name 		= os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, "Data", "SubjectData", "OptDimDict.json"))

# funciton for flattening lists
flatten 		= lambda l: [item for sublist in l for item in sublist]


def col_names_tuple(n_features):
    row1_colnames = [ ["val"]*(n_features + 2), ["test"]*(n_features + 2) ]
    row2_colnames = [ ["X"]*n_features, ["y_c4"], ["y_c6"] ]*2
    row3_colnames = [range(n_features), [1]*2] * 2
    
    col_names = map(flatten,[ row1_colnames, row2_colnames, row3_colnames])
    
    return list(zip(*col_names))

def col_recon_tuple():
    n_features = 1025
    row1_colnames = [ ["val"]*(n_features), ["test"]*(n_features) ]
    row2_colnames = [range(n_features * 2)]
    
    col_names = map(flatten,[ row1_colnames, row2_colnames])
    
    return list(zip(*col_names))



def compute_nmf_logLik(subject_spec1,nmf_recon):
    
    [r, c] = subject_spec1.shape

    diffSq = (nmf_recon - subject_spec1)**2
    sigma = np.sum(diffSq) / (r*c)

    # inverse variance
    beta = 1 / sigma

    negLoglike = - beta * np.sum(diffSq)/2 + r * c * np.log(beta) / 2 - r * c * np.log(2 * np.pi) / 2
    return negLoglike
    


def find_optimum_nmf_dims(subject_spec1, subject_name, dim_values):
    
    lowest_BIC = np.inf
    opt_dim = 0
    
    [r, c] = subject_spec1.shape
    for n in dim_values:
        model = NMF(n_components=n, init='nndsvd', random_state=0)
        W1 = model.fit_transform(subject_spec1)
        nmf_recon = model.inverse_transform(W1)
        
        negLoglike = compute_nmf_logLik(subject_spec1, nmf_recon)
        Q = n*(r + c)
        
        current_BIC = -2 * negLoglike + 2 * Q * np.log(r * c) / 2;
        
        if current_BIC < lowest_BIC:
            lowest_BIC = current_BIC
            opt_dim = n
            best_model = model
    logging.debug("[Worker %d] Optimal dimensionality for subject %s: %i" % (os.getpid(), subject_name, opt_dim))
    return {"nmf_dim" : opt_dim, "nmf_BIC": lowest_BIC}, best_model

def work_best_dim(work_tuple):

	d, model, x1, y1_4, y1_6, x2, y2_4, y2_6, nmf_file_name, reconstruct_file_name, subject_name = work_tuple
	n_features 					= d["nmf_dim"]
	
	xtrain						= model.transform(x1)
	spec_recon_train 				= model.inverse_transform(xtrain)
	xtest 						= model.transform(x2)
	spec_recon_test 				= model.inverse_transform(xtest)
	data_recon 					= pd.concat([pd.DataFrame(spec) for spec in [spec_recon_train, spec_recon_test]], axis=1, ignore_index=True)
	data 						= pd.concat([pd.DataFrame(xtrain),pd.DataFrame(y1_4),pd.DataFrame(y1_6), pd.DataFrame(xtest), pd.DataFrame(y2_4), pd.DataFrame(y2_6)], axis=1, ignore_index = True)
	
	# setting pandas columns
	index 						= pd.MultiIndex.from_tuples(col_names_tuple(n_features), names = ['sets', 'dataspec', 'columns'])
	data.columns 					= index
	index_recon 					= pd.MultiIndex.from_tuples(col_recon_tuple(), names=['sets', 'columns'])
	data_recon.columns 				= index_recon
	
	# pickle data structures
	data.to_pickle(nmf_file_name, compression = 'gzip')	    
	data_recon.to_pickle(reconstruct_file_name, compression = 'gzip')
	
def main(dim_values):
	# import environment which contains number of available cores
	try:
		n_cores			= int(os.environ['PBS_NUM_PPN'])
	except Exception as e:
		logging.error(e)
		n_cores			= -1
	logging.info("Reading data from .mat file")
	timer		= time.time()
	
	SpecsLabels 	= sio.loadmat(mat_file_name)
	
	dt		= time.time() - timer
	# time.gmtime converts seconds to correct format for strftime.
	dt 		= time.strftime("%H-%M-%S", time.gmtime(dt))
	logging.info("Finished reading .mat file. Elapsed time: {0}".format(dt))
	
	
	num_subjects 	= len(SpecsLabels['SPEC_1'][0])
	subject			= ['subject_' + str(num) for num in range(num_subjects)]
	
	X_train		= SpecsLabels['SPEC_1'][0]
	zipped_data	= zip(SpecsLabels['SPEC_1'][0], SpecsLabels['ANNOT_1'][0],SpecsLabels['ANNOTORIG_1'][0], SpecsLabels['SPEC_2'][0],SpecsLabels['ANNOT_2'][0],SpecsLabels['ANNOTORIG_2'][0], nmf_file_name, reconstruct_file_name, subject)
	
	
	try:
		# create a pool of workers using subprocesses as backend
		# this pool is then reused for both finding nmf and exporting files
		
		with Parallel(n_jobs = n_cores, verbose = 10) as parallel:
			logging.info("Beginning parallel decomposition in order to find optimum dimensionality")
			timer		= time.time()
			train_results 		= parallel(delayed(find_optimum_nmf_dims)(x_train, subject_name, dim_values) for x_train, subject_name in zip(X_train, subject))
			zipped_data		= zip(train_results, zipped_data)
			
			dt		= time.time() - timer
			dt 		= time.strftime("%H-%M-%S", time.gmtime(dt))
			logging.info("Finished finding optimal dimensionality for all subjects. Elapsed time: {0}".format(dt))
			logging.info("Reconstructing spectra and exporting demensionality reduced data.")
			
			timer		= time.time()
			parallel(delayed(work_best_dim)(result_tuple + data_tuple) for result_tuple, data_tuple in zipped_data)
			dt		= time.time() - timer
			dt 		= time.strftime("%H-%M-%S", time.gmtime(dt))
			logging.info("Finished exportating data. Elapsed time: {0}".format(dt))
		
		
		# create final result dictionary and exporting it
		opt_n_features_list, _ 	= zip(*train_results)	
		opt_n_features 	= {sub: d for sub, d in zip(subject, opt_n_features_list)}
		
		with open(dict_file_name, 'w') as outfile:
			json.dump(opt_n_features, outfile)
	except:
		PrintException()
	
	
	
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--dim_min", type = int,  help = "Integer specifying minimum dimension", default = 10)
	
	parser.add_argument("--dim_interval", type = int, help = "Integer specifying dimension interval to iterator over" , default = 60)
	
	parser.add_argument("--dim_max", type = int, help = "Integer specifying maximum dimension" , default = 100)
	
	
	args 		= parser.parse_args()


	logging.basicConfig(format = '%(asctime)s, %(levelname)s, %(name)s, -- %(message)s',
			    datefmt = "%m-%d %H:%M:%S",
			    level = logging.INFO)
    	logging.info("Starting the nmf decomposition of all subjects")
    	dim_values	= range(args.dim_min, args.dim_max, args.dim_interval)
	main(dim_values)
