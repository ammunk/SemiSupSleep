import pandas as pd
import os
import numpy as np

# using __file__ provides current directory
current_dir = os.path.dirname(__file__)
#os.path.join uses the appropriate system child folder specification (i.e. "/,\,..")
file_name = [os.path.abspath(os.path.join(current_dir, os.pardir,os.pardir, "Data","SubjectData","NMFandLabelsSubject_" + str(sub)+ ".gzip")) for sub in range(19)]

def col_names_tuple(n_features):
	row1_colnames = [ ["val"]*(n_features + 2), ["test"]*(n_features + 2) ]
	row2_colnames = [ ["X"]*n_features, ["y_c4"], ["y_c6"] ]*2
	row3_colnames = [range(n_features), [1]*2] * 2

	col_names = map(flatten,[ row1_colnames, row2_colnames, row3_colnames])
    
	return list(zip(*col_names))

flatten = lambda l: [item for sublist in l for item in sublist]

def get_three_clusters(funnytest=False):
    
	# Specify clusters:
	N_1 = 200
	N_2 = 100
	N_3 = 50

	N_labels_prct = 10
	wrongLabelFrac = 2.0/3.0

	mean1 = np.array([5, 5])
	mean2 = -mean1
	mean3 = np.array([-5, 5])

	cov1 = [[1.2, 0.5], [0.5, 1]]
	cov2 = [[1, -0.5], [-0.5, 1]]
	cov3 = [[0.9, 0.3], [0.3, 1]]

	# -------------------------------------------------------------------------
	# Make first cluster
	x1, y1 = np.random.multivariate_normal(mean1, cov1, N_1).T
	x2, y2 = np.random.multivariate_normal(mean2, cov2, N_2).T
	x3, y3 = np.random.multivariate_normal(mean3, cov3, N_3).T

	ntrue1label1 = round(N_1*wrongLabelFrac)
	ntrue1label2 = round((N_1*(1-wrongLabelFrac))*N_2/(N_2 + N_3))
	ntrue1label3 = N_1 - ntrue1label1 - ntrue1label2

	true1labels1 = np.ones([int(ntrue1label1),1],dtype=np.int8)
	true1labels2 = np.ones([int(ntrue1label2),1],dtype=np.int8)*2
	true1labels3 = np.ones([int(ntrue1label3),1],dtype=np.int8)*3 
	labels1 = np.concatenate((true1labels1,true1labels2,true1labels3))
	if funnytest:
		labels1 = np.ones([N_1,1],dtype=np.int8)
		labels1[0] = 2
		labels1[1] = 3
            
	# -------------------------------------------------------------------------
	# Make second cluster
	ntrue2label2 = round(N_2*wrongLabelFrac)
	ntrue2label1 = round((N_2*(1-wrongLabelFrac))*N_1/(N_1 + N_3))
	ntrue2label3 = N_2 - ntrue2label1 - ntrue2label2

	true2labels1 = np.ones([int(ntrue2label1),1],dtype=np.int8)
	true2labels2 = np.ones([int(ntrue2label2),1],dtype=np.int8)*2
	true2labels3 = np.ones([int(ntrue2label3),1],dtype=np.int8)*3    
	labels2 = np.concatenate((true2labels1,true2labels2,true2labels3))

	# -------------------------------------------------------------------------
	# Make third cluster
	ntrue3label3 = round(N_3*wrongLabelFrac)
	ntrue3label1 = round((N_3*(1-wrongLabelFrac))*N_1/(N_1 + N_2))
	ntrue3label2 = N_3 - ntrue3label1 - ntrue3label3

	true3labels1 = np.ones([int(ntrue3label1),1],dtype=np.int8)
	true3labels2 = np.ones([int(ntrue3label2),1],dtype=np.int8)*2
	true3labels3 = np.ones([int(ntrue3label3),1],dtype=np.int8)*3    
	labels3 = np.concatenate((true3labels1,true3labels2,true3labels3))

	X = np.expand_dims(np.concatenate((x1,x2,x3)),1)
	Y = np.expand_dims(np.concatenate((y1,y2,y3)),1)
	L = np.concatenate((labels1,labels2,labels3))

	dataset = np.concatenate((X,Y,L,L),axis=1)
	return dataset

def get_two_clusters():
    
	# Specify clusters:
	N_1 = 200
	N_2 = 200

	wrongLabelFrac = 2.0/3.0

	mean1 = np.array([3, 3])
	mean2 = -mean1

	cov1 = [[1.2, 0.5], [0.5, 1]]
	cov2 = [[1, -0.5], [-0.5, 1]]


	x1, y1 = np.random.multivariate_normal(mean1, cov1, N_1).T
	x2, y2 = np.random.multivariate_normal(mean2, cov2, N_2).T
	# -------------------------------------------------------------------------
	# Make labels for first cluster

	ntrue1label1 = round(N_1*wrongLabelFrac)
	ntrue1label2 = N_1 - ntrue1label1

	true1labels1 = np.ones([int(ntrue1label1),1],dtype=np.int8)
	true1labels2 = np.ones([int(ntrue1label2),1],dtype=np.int8)*2
	labels1 = np.concatenate((true1labels1,true1labels2))

	# -------------------------------------------------------------------------
	# Make labels for second cluster
	ntrue2label2 = round(N_2*wrongLabelFrac)
	ntrue2label1 = N_2 - ntrue2label2

	true2labels1 = np.ones([int(ntrue2label1),1],dtype=np.int8)
	true2labels2 = np.ones([int(ntrue2label2),1],dtype=np.int8)*2
	labels2 = np.concatenate((true2labels1,true2labels2))

	# -------------------------------------------------------------------------

	X = np.expand_dims(np.concatenate((x1,x2)),1)
	Y = np.expand_dims(np.concatenate((y1,y2)),1)
	L = np.concatenate((labels1,labels2))

	dataset = np.concatenate((X,Y,L,L),axis=1)
	return dataset

def get_synthetic_dataset():
	trainset = get_three_clusters()
	testset  = get_three_clusters()

	#trainset = get_two_clusters()
	#testset  = get_two_clusters()

	dataset = pd.concat([pd.DataFrame(trainset), pd.DataFrame(testset)], axis=1, ignore_index = True)

	index = pd.MultiIndex.from_tuples(col_names_tuple(2), names = ['sets', 'dataspec', 'columns'])
	dataset.columns = index

	return dataset

def standardize_data(data):
	n_features = data['test']['X'].shape[1]
	normVal = data['val']['X']/data['val']['X'].quantile(0.99)
	normTest = data['test']['X']/data['val']['X'].quantile(0.99)
	normdata = pd.concat((normVal,data['val']['y_c4'],data['val']['y_c6'], normTest, data['test']['y_c4'], data['test']['y_c6']), axis=1, ignore_index = True)

	# setting pandas columns
	index = pd.MultiIndex.from_tuples(col_names_tuple(n_features), names = ['sets', 'dataspec', 'columns'])

	normdata.columns = index
	return normdata

# define generator (which can be looped over)	
def subject_generator(test_run):
	
	if test_run:
		data = get_synthetic_dataset()
		yield data['val'], data['test'], 'SYNTH_DATA'
	else:
		# test_run specifies if only one subject should be analyzed (for testing of algorithm)
		subjects	= ['subject_' + str(num) for num in range(19)]
		for subject, path in zip(subjects, file_name):
			data 	= pd.read_pickle(path, compression = 'gzip')
			data	= standardize_data(data)
			yield data['val'], data['test'], subject

