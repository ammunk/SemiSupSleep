from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV as GSCV
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import logging, sys, os
from tqdm import tqdm
from data_type import to_numpy, remove_nan, y_c4, y_c6, SYNTH, SYNTH_3
from sklearn.metrics import confusion_matrix as confusion_Mat
from scipy import linalg
from sklearn.cluster import KMeans
from scipy.misc import logsumexp
from sklearn.metrics import fbeta_score, make_scorer
import linecache


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    logging.critical('EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj))

label_enum = {'4': y_c4, '6': y_c6, '2': SYNTH, '3': SYNTH_3}
# ignore all warnings associated with floating point operations such as e.g. np.log(1e-20000).
# We can ignore this because it is associated with 0*log(0) = 0, but because np.log(0)=-inf we get 0(-inf) = nan due to numpy, we later in the code handle these special cases. In order not to flood the stderr, we suppress such warnings.
np.seterr(all='ignore')
# remember numpy distinguishes between dimension N*M*..., N*1 or just N... I.e. dimensionality >2, 2 or 1.
# one dimensional arrays (when doing broadcasting for elementwise operations) are considered to be along the first axis of the "matrix" (due to zero indexing) - along x axis
#	otherwise one dimensional arrays are "transposed" into the appropriate dimensionality, when using eg. numpy.dot(a,b)

# Also np operations work on pandas dataframe
import numpy as np



def validation_and_testing(estimator, parameters, skf_val):
	# cls is the model class to undertake testing/validation
	# init_dict_args is a dictionary of arguements passed as the initialization (data) of cls
	# parameters: on which to perform gridsearch
	# kwargs are whatever other namegiven arguments passed

	
	# import environment which contains number of available cores
	try:
		n_cores			= int(os.environ['PBS_NUM_PPN'])
	except:
		n_cores			= -1

	# parameters are bassed to estimator, by doing deep_copying. Hence, the original parameters initialized to the estimator (init_dict_args, which are not model specific (i.e. replaced by "parameters")) will remain set.
	return GSCV(estimator, parameters, n_jobs = n_cores, cv = skf_val, return_train_score = False, verbose = 0)
		
####################################################################################

# parameter estimations

def _means(X, resp, nk):
	n_components	= len(nk)
	# use broadcasting
	means = np.sum(resp.T.reshape(n_components, 1, -1) * X.T, axis = 2).T / nk
	return means.T
	
def _covariance_full(X, resp, means, nk):
	n_components, n_features 	= means.shape
	
	covariances			= np.empty((n_components, n_features, n_features))
	for k in range(n_components):
		diff			= X - means[k]

		covariances[k]		= np.dot(resp[:,k] * diff.T, diff) / nk[k]
	return covariances

def _covariance_full_tied(X, resp, means, nk):
	n_components, n_features 	= means.shape
	covariances			= np.empty((n_features, n_features))

	avg_X2 = np.dot(X.T, X)
	avg_means2 = np.dot(nk * means.T, means)
	covariance = avg_X2 - avg_means2
	covariance /= nk.sum()
	return covariance
    
def _covariance_diag(X, resp, means, nk):   
	n_components, n_features 	= means.shape
	covariances			= np.empty((n_components, n_features))

	#Using summation trick see notes/pictures
	avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
	avg_means2 = means ** 2
	covariances = avg_X2 - avg_means2
	return covariances

def _covariance_diag_tied(X, resp, means, nk):   
	n_components, n_features 	= means.shape
	covariances			= np.empty((n_features))

	avg_X2 = np.sum(X ** 2, axis = 0)
	avg_means2 = np.dot(nk, means**2)
	covariances = (avg_X2 - avg_means2)
	covariances /= nk.sum()

	return covariances

def _covariance_spherical(X, resp, means, nk):
    
	return _covariance_diag(X, resp, means, nk).mean(1)

def _covariance_spherical_tied(X, resp, means, nk):
    
	return _covariance_diag_tied(X, resp, means, nk).mean()

def _components(n_samples, nk):
	return nk / n_samples
	
def _class_prob(resp, class_resp, y, label_idx, unlabel_independent):
	n_samples, _ = resp.shape
	if (n_samples == y.shape[0] or unlabel_independent):
		# in this case class_resp is simply an n_classes by n_components
		n_classes, n_components	= class_resp.shape
		p_ak			= np.zeros((n_classes, n_components))
		normalizer		= np.sum(resp[label_idx, :], axis = 0)
	else:
		p_ak 			= np.sum(class_resp, axis = 0)
		normalizer		= np.sum(resp, axis = 0)

	for idx, c in enumerate(y):	
		p_ak[int(c),:]	+= resp[label_idx[idx], :]

	return (p_ak / normalizer)
	
def _estimate_parameters(X, y, resp, nk, class_resp, cov_type, reg, label_idx, unlabel_independent):
	n_samples, n_features	= X.shape
	components		= _components(n_samples, nk)
	means			= _means(X, resp, nk)
	if np.any(np.isnan(means)):
		raise ValueError("Collapsing probabilities - one components has all zero responsibilities")
 	if cov_type == "full":
		covariances	= _covariance_full(X, resp, means, nk)
	elif cov_type == "full_tied":
		covariances	= _covariance_full_tied(X, resp, means, nk)
    	elif cov_type == "diag":
		covariances	= _covariance_diag(X, resp, means, nk)
    	elif cov_type == "diag_tied":
		covariances	= _covariance_diag_tied(X, resp, means, nk)
   	elif cov_type == "spherical":
		covariances	= _covariance_spherical(X, resp, means, nk)
    	elif cov_type == "spherical_tied":
		covariances	= _covariance_spherical_tied(X, resp, means, nk)
	if reg:
		if 'full' in cov_type:
			covariances += np.diag(np.zeros((n_features,1)) + reg)
		else:
			covariances += reg
		
	precisions_chol = _cholesky_factorization_precision(covariances, cov_type)

	p_ak		= _class_prob(resp, class_resp, y, label_idx, unlabel_independent)

	return components, means, covariances, precisions_chol, p_ak

####################################################################################

# cholesky utilized functions
	
def _cholesky_factorization_precision(covariances, cov_type):
	
	if cov_type == 'full':
		n_components, n_features, _ 	= covariances.shape
		precisions_chol			= np.empty((n_components, n_features, n_features))
		for k in range(n_components):
			try:
				cov_chol	= linalg.cholesky(covariances[k], lower = True)
			except linalg.LinAlgError:
		        	raise ValueError("Ill-defined covariance matrix. Try another covariance type or add a regularization term in order to alleviate the issue")

			# return transposed of the inverse of the lower triangular matrix, as this need to be used smartly for the normal_log_prob. Note, det(A.T) = det(A)
		    	precisions_chol[k]	= linalg.solve_triangular(cov_chol, np.identity(n_features), lower = True).T
	elif cov_type == 'full_tied':
		n_features, _ 	= covariances.shape
		try:
			cov_chol = linalg.cholesky(covariances, lower = True)
		except linalg.LinAlgError:
			raise ValueError("Ill-defined covariance matrix. Try another covariance type or add a regularization term in order to alleviate the issue")
		# return transposed of the inverse of the lower triangular matrix, as this need to be used smartly for the normal_log_prob. Note, det(A.T) = det(A) 
		precisions_chol = linalg.solve_triangular(cov_chol, np.identity(n_features), lower = True).T
	else:
		cov_chol = 1. / np.sqrt(covariances)	
		if np.isinf(cov_chol).any():
			raise ValueError("Ill-defined covariance matrix. Try another covariance type or add a regularization term in order to alleviate the issue")
		precisions_chol = cov_chol			

	return precisions_chol
	
def _log_det_chol(lower_tri_mat, cov_type, n_features):
	# use that the cholesky factorization is lower triangular, hence the determinant is the product of the diagonal!    
    	# https://proofwiki.org/wiki/Determinant_of_Triangular_Matrix    
    	# remember det(Sigma^-1) = det(L * L.T) = det(L) ^ 2
	if cov_type == "full":  
		n_components, _, _ =  lower_tri_mat.shape    
		return np.sum(np.log(lower_tri_mat.reshape((n_components,-1))[:, ::n_features + 1]), axis = 1) #Along 2. axis
	elif cov_type == "full_tied":
		return np.sum(np.log(np.diag(lower_tri_mat)))
	elif cov_type == 'diag':
		return np.sum(np.log(lower_tri_mat), axis = 1) 
	elif cov_type == 'diag_tied':
		return np.sum(np.log(lower_tri_mat))
	else:
		return n_features * np.log(lower_tri_mat)

####################################################################################

# log probabilities and responsibilities
	
def _log_prob_gauss(X, means, precisions_chol, cov_type):
	
	n_samples, n_features	= X.shape
    	n_components, _		= means.shape
    
    # we use we have the precision decomposition, ie. L^-1, such that sqrt(1/det(Sigma)) = sqrt(det(Sigma^-1)), which is easily calculated using the decomposition
    	log_det			= _log_det_chol(precisions_chol, cov_type, n_features)
    
    	if cov_type == "full":
		log_exp		= np.empty((n_samples, n_components))
		for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
		    # Use the cholesky factorization to realize we end up with a simple dot product of the same vector for each sample n - note prec_chol = (L.T)^-1
		    diff		= np.dot(X, prec_chol) - np.dot(mu,prec_chol) # N x D 
		    log_exp[:, k] 	= np.sum(np.square(diff), axis = 1) #N x 1
		    
    	elif cov_type == "full_tied":
        	log_exp		= np.empty((n_samples, n_components))
        	for k, (mu) in enumerate(means):
		    # Use the cholesky factorization to realize we end up with a simple dot product of the same vector for each sample n - note prec_chol = (L.T)^-1
		    diff		= np.dot(X, precisions_chol) - np.dot(mu,precisions_chol) # N x D 
		    log_exp[:, k] 	= np.sum(np.square(diff), axis = 1) #N x 1
		    
    	elif cov_type == "diag" or cov_type == "diag_tied":   
		precision_sq 	= precisions_chol ** 2
		
		if "tied" in cov_type:
			precision_sq	= np.tile(precision_sq, (n_components, 1))
		
		log_exp 	= (np.sum((means ** 2 * precision_sq), 1) - 2. * np.dot(X, (means * precision_sq).T) + np.dot(X ** 2, precision_sq.T))
   	
   	elif cov_type == "spherical":
		precision_sq	= precisions_chol ** 2
		X_norm 		= np.sum(X ** 2, axis = 1)
		log_exp 	= np.sum(means ** 2, 1) * precision_sq - 2 * np.dot(X, means.T * precision_sq) + np.outer(X_norm, precision_sq)
   	
   	elif cov_type == "spherical_tied":
   		
		precision_sq 	= precisions_chol ** 2
		X_norm 		= np.sum(X ** 2, axis = 1)
		log_exp 	= (np.sum(means ** 2, 1) * precision_sq - 2 * np.dot(X, means.T * precision_sq)).T + X_norm * precision_sq
		log_exp		= log_exp.T

	# utilize the broadcasting, which sums along equal size dimension (i.e. n_components)
	# also use that the determinant of the inverse is one over the determinant of non-inverse
	# the log (again using cholesky) takes the 0.5 factor from the last term
	log_prob_gauss		= - 0.5 * (n_features * np.log(2 * np.pi) + log_exp) + log_det
    	
	return log_prob_gauss

def _loglik(log_prob_gauss, resp, p_ak, components, y = None, class_resp = None, unlabel_idx = None, label_idx = None):
	
	n_classes, _ = p_ak.shape
	
	if y is not None:
		# pick out the p_ak for known labels
		tmp   = np.asarray([p_ak[int(c), :] for c in y])
		loglik = resp[label_idx] * (log_prob_gauss + np.log(components) + np.log(tmp))
	else:
		if class_resp.ndim ==  3:
			loglik = resp[unlabel_idx] * (log_prob_gauss + np.log(components)) + np.sum(class_resp * np.log(p_ak), axis = 1)
		else:
			loglik = resp[unlabel_idx] * (log_prob_gauss + np.log(components))
	
	if np.any(np.isnan(loglik)):
		# set all 0*log(0) = 0 which otherwise results to nan
		# would happen both if p(y|k) = 0 or p(x|k) = 0
		loglik = np.nan_to_num(loglik)
	# return loglik for each observation (i.e. summed across components)
	return np.sum(loglik, axis = 1)
	
def _log_resp(log_prob_gauss, components):
	
	log_resp		= log_prob_gauss + np.log(components)
	
	# subtract largest value (in exp argument - ie. directly from the log) to alleviate overflow
	# the transpose handles the broadcasting as np.sum makes it 1D
	log_resp		= log_resp.T - log_resp.max(axis = 1)
	log_resp		= log_resp  -  logsumexp(log_resp.T, axis = 1)
	# return the retransposed
	return log_resp.T


def get_prior(y):
	# y is one dimensional, hence size is appropriate
	n_samples	= y.size
	uniques, counts = np.unique(y, return_counts = True)
	return counts / float(n_samples)

def loss_mat(prior):
	n_classes	= prior.size
	L		= np.zeros((n_classes, n_classes))
	for i, pred in enumerate(prior[:-1]):
		for j, true in enumerate(prior[i+1:]):
			j = j + i + 1
			ratio = pred/true
			if ratio > 1:
				L[i,j] = ratio
				L[j,i] = 1
			else:
				L[j,i] = 1/ratio
				L[i,j] = 1
	return L
			
#####################################################################################


class ClusterClass(BaseEstimator, ClassifierMixin):
	"""
	 inherit from sklearn (BaseEstimator) - to make it sklearn compatible!, see - http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
	
	 the above inheritance allows for eg set_params and get_params, which would be set in super class __init__(), ie. BaseEstimator 
	
	 the scoring method is inherited from "ClassifierMixin"
	
	 attributes which are public AND set in fit() should have trailing "_"
	
	 This class is meant to be called when doing the cross validation, such that we do not need to reprovide the data, for every time we need to train a new GMM.
	
	 CONSIDER IMPLEMENTING THIS AS A THREAD!!!!
	
	"""
	
	# parameters to be shared across all instances of the class

	cov_type 		= ""
	n_init 			= ""
	max_iter		= ""
	tolerance		= ""
	unlabel_independent_	= ""
	reg			= ""
	prior_loss_on		= ""
	
	def __init__(self, n_components_ = 'n_components', unlabel_weight = 'unlabel_weight', reg = None):
		self.n_components_ 	= n_components_
		self.unlabel_weight	= unlabel_weight
		
	# this method is called by GridSearchCV, and fits the model
	def fit(self, X, y, frac = None):
		
		X, y						= check_X_y(X, y)
		# zero index the classes
		y						= y - 1	

		self.n_samples, self.n_features			= X.shape
		
		# check if a frac is specified, otherwise reuse earlier provided frac
		if frac:
			self.frac = frac
		
		# choose all indecies per default
		self.unlabel_idx, self.label_idx		= None, range(self.n_samples)
		self.n_unlabel, self.n_labels			= 0, self.n_samples
		
		if self.frac < 1:
			sss					= StratifiedShuffleSplit(n_splits=1, test_size = self.frac)
		
						# sss.split returns an iterator (using yield), hence the use of next()
			idx_iterator				= sss.split(np.empty(self.n_samples), y)
		
			
			self.unlabel_idx, self.label_idx	= idx_iterator.next()
			self.n_unlabel, self.n_labels		= map(len, [self.unlabel_idx, self.label_idx])

		y_label	= y[self.label_idx]
		self.classes_					= unique_labels(y)
		self.n_classes_					= len(self.classes_)
		self.prior					= get_prior(y_label)
		# max_lower_log_prob allows for comparisons of several initializations in order to pick best one
		max_lower_log_prob				= -np.infty
		
		for _ in range(self.n_init):
			# initialize the algorithm for n_init times = the only initialization difference if the means calculated by k-means
			self._initialize(X)

			# calculate the first log_prob_gauss used in the e-step for the responsibilities
			self.log_prob_gauss				= _log_prob_gauss(X, self.means, _cholesky_factorization_precision(self.covariances, self.cov_type), self.cov_type)
			
			loglik_semi_supervised				= -np.infty
			converged					= False
			for i in range(self.max_iter):
					lower_log_prob				= loglik_semi_supervised
					
					self._e_step(y_label)
					self._m_step(X, y_label)
					loglik_semi_supervised			= self._loglik_semi_supervised(X, y_label)
					loglik_diff				= loglik_semi_supervised - lower_log_prob
					
					if loglik_diff < self.tolerance * abs(loglik_semi_supervised) and loglik_diff > 0:
						converged 			= True
						break
			if loglik_semi_supervised > max_lower_log_prob:
				max_lower_log_prob 		= loglik_semi_supervised
				means				= self.means
				covariances			= self.covariances
				components			= self.components
				p_ak				= self.p_ak
				
				self.converged_ 		= converged
				self.iterations_ 		= i

		self.means			= means
		self.covariances		= covariances
		self.components			= components
		self.p_ak			= p_ak
		self.X_				= X
		self.y_				= y + 1
		# to allow chaining - probably used by sklearn, needed as by documentation
		return self
	
	def _initialize(self, X):

		# if unlabeled data is considered not having latent class variables: initilialize to be zero and code compatible. Should also happen if we have no unlabels - code compatibility
		if self.unlabel_idx is None or self.unlabel_independent_:		
			self.class_resp		= np.zeros((self.n_classes_, self.n_components_))
		
		# be careful when using np.empty... see documentation. np.zeros may be better
		self.p_ak			= np.zeros((self.n_classes_, self.n_components_)) + self.prior.reshape(self.n_classes_,1)

		if self.cov_type == "full":
			self.covariances	= np.tile(np.cov(X, rowvar = False), (self.n_components_, 1, 1))
		elif self.cov_type == "full_tied":
			self.covariances	= np.cov(X, rowvar = False)
		    
		elif self.cov_type == "diag":
			self.covariances	= np.tile(np.diag(np.cov(X, rowvar = False)), (self.n_components_, 1))
		    
		elif self.cov_type == "diag_tied":
			self.covariances	= np.diag(np.cov(X, rowvar = False))
		    
		elif self.cov_type == "spherical":
			m = (np.sum(X, axis = 0)/self.n_features)
			dist = np.sqrt(np.sum(np.square(X-m), axis = 1))
			self.covariances	= np.tile(np.average(dist), (self.n_components_))
		    
		elif self.cov_type == "spherical_tied":
			m = (np.sum(X, axis = 0)/self.n_features)
			dist = np.sqrt(np.sum(np.square(X-m), axis = 1))
			self.covariances	= np.average(dist)
		
		self.components		= 1.0 / self.n_components_
		
		
		# we use k-means++ as initialization for the k-means (default)
		means = KMeans(n_clusters = self.n_components_, n_init=10).fit(X).cluster_centers_
		self.means = np.asarray(means)
	
	def _e_step(self, y,):

		  
     		self.resp					= np.exp(_log_resp(self.log_prob_gauss, self.components))
     		# Overwrite labelled resp, to account for cluster-class relation
		for idx, c in enumerate(y):
			norm                			= np.sum(self.p_ak[int(c),:] * 	self.resp[self.label_idx[idx], :])
			self.resp[self.label_idx[idx], :]     	*= (self.p_ak[int(c),:] / norm)		
		
		# multiply unlabeled resposibilities with unlabel_weight if we have unlabeled data
		if not self.unlabel_idx is None:
			self.resp[self.unlabel_idx] *= self.unlabel_weight
		
		# class_resp is only as long as number of unlabels: resp*p_ak dimenions (n x a x k))
		# check whether unlabel data is considered to have latent class variables OR
		# unlabeled data even exists
		if not (self.unlabel_idx is None or self.unlabel_independent_):
			self.class_resp	= self.resp[self.unlabel_idx, :].reshape(self.n_unlabel, 1, -1) * self.p_ak
				
	def _m_step(self, X, y):
		nk 		= np.sum(self.resp, axis = 0)

		self.components, self.means, self.covariances, self.precisions_chol, self.p_ak 	= _estimate_parameters(X, y, self.resp, nk, self.class_resp, self.cov_type, self.reg, self.label_idx, self.unlabel_independent_)
		
		if np.any(np.all(np.isnan(self.p_ak),axis = 0)):
			# find those components, which we could not assign to a class
			# Those columns contains NaN, which we replace by class priors
			nan_cols			= np.where(np.all(np.isnan(self.p_ak),axis = 0))
			for nan_col in nan_cols[0]:
				self.p_ak[:, nan_col] = self.prior
	
	def _loglik_semi_supervised(self, X, y):
		self.log_prob_gauss		= _log_prob_gauss(X, self.means, self.precisions_chol, self.cov_type)
		
		loglik_label 			= _loglik(self.log_prob_gauss[self.label_idx,:], self.resp, self.p_ak, self.components, y = y, label_idx = self.label_idx)
	
		# then do a logsumexp along axis = 1, then sum over axis = 0
		# check if there exists unlabeled data
		if not (self.unlabel_idx is None):
			loglik_unlabel 		= _loglik(self.log_prob_gauss[self.unlabel_idx,:], self.resp, self.p_ak, self.components, class_resp = self.class_resp, unlabel_idx=self.unlabel_idx)	
		else:
			loglik_unlabel 		= 0
		
		
		# return loglik summed across observations
		loglik_semi_supervised 	= np.sum(loglik_label,axis=0) + np.sum(loglik_unlabel, axis = 0)
		return loglik_semi_supervised
	
			
	def predict(self, X):
		check_is_fitted(self, ['X_', 'y_'])
		n_samples, _		= X.shape
		log_prob_gauss		= _log_prob_gauss(X, self.means, self.precisions_chol, self.cov_type)
		resp			= np.exp(_log_resp(log_prob_gauss, self.components))
		
		class_prob		= np.sum(resp.reshape(n_samples, 1, -1) * self.p_ak, axis = 2)
		if self.prior_loss_on:
			L		= loss_mat(self.prior)
			class_prob	= -np.dot(class_prob,L)
			
		# add one, to one make it one-indexed
		return class_prob.argmax(axis = 1) + 1
			
	def test_score(self, d_val, d_test):
		self.fit(*d_val)
		return self.score(*d_test)
	
	def get_parameters(self):
		# return parameters as dictionaries
		mean_dict 	= {'component_' + str(component): mean.tolist() for component, mean in zip(range(1,self.n_components_ + 1), self.means)}
		
		if "tied" not in self.cov_type:
			cov_dict	= {'component_' + str(component): cov.tolist() for component, cov in zip(range(1,self.n_components_ + 1), self.covariances)}
		else:
			cov_dict = {'tied_covariance': self.covariances.tolist()}
		return mean_dict, cov_dict
	
	def confMat(self, label_type, X, y):
		y_pred			= self.predict(X)
		cm			= confusion_Mat(y, y_pred)
		cm 			= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	    	cm_pd 			= pd.DataFrame(cm)
		
		n_labels		= cm_pd.shape[0]
		col_lab			= ["Pred"] * n_labels
		idx_lab			= ["True"] * n_labels
		
		[col_index, row_index] 	=  [ pd.MultiIndex.from_tuples(idx, names=['Labels', 'Class']) for idx in [zip(lab, label_enum[str(n_labels)].sleep_types()) for lab in [col_lab, idx_lab]]]
	    	cm_pd.columns 	= col_index
	    	cm_pd.index 	= row_index
	    	
	    	return cm_pd.to_json(orient='index')

	@classmethod	
	def update_class_vars(cls, **kwargs):
		# consider if all kwargs should somehow be used
		cls.cov_type 			= kwargs['cov_type']
		cls.n_init 			= kwargs['n_init']
		cls.max_iter			= kwargs['max_iter']
		cls.tolerance			= kwargs['tolerance']
		cls.unlabel_independent_	= kwargs['unlabel_independent']
		cls.reg				= kwargs['reg_term']
		cls.prior_loss_on		= kwargs['prior_loss_on']

def frac_model(estimator, d_val, d_test, skf_val, parameters, label_type, frac, progressbar):
	sleep_gridsearch = validation_and_testing(estimator, parameters, skf_val)
	
	d_val, d_test	 = to_numpy(d_val, d_test, label_type)
	X, y 		 = d_val
	sleep_gridsearch.fit(*d_val, frac = frac)
	best_estimator				= sleep_gridsearch.best_estimator_
	test_score				= best_estimator.test_score(d_val, d_test)
	means, covariances			= best_estimator.get_parameters()
	conf_mat				= best_estimator.confMat(label_type, *d_test)
	iterations				= best_estimator.iterations_
	progressbar.update(1)
	return {'best_val_score': sleep_gridsearch.best_score_, 'test_score': test_score, 'best_params': sleep_gridsearch.best_params_, 'means': means, 'covariances': covariances,  'confusion_matrix': conf_mat, "n_iterations_best_model": iterations}
	

def model_generator(parsed, parameters, d_val, d_test, subject_name):

	label_type		= parsed['label_type']
	n_split_val		= parsed['n_split']
	enable_progressbar	= parsed['enable_progress_bar']
	skf_val 		= StratifiedKFold(n_splits=n_split_val, shuffle = True)
	logger			= logging.getLogger(__name__)
	d_val, d_test		= remove_nan(d_val, d_test)
	# initialize object
	estimator		= ClusterClass()
	# update class object
	estimator.update_class_vars(**parsed)
	devide			= lambda x: x/float(100)
	interval		= 10
	frac_init		= interval * 2 # start at 20%
	up_lim			= 100 + interval
	
	
	
	try:
		if up_lim % interval != 0:
			raise Exception('Issues with the frac range')
		frac_range	= map(devide, range(frac_init, up_lim, interval))

	except:
		PrintException()
		sys.exit(1)
	
	logger.info('Working on {0} \n'.format(subject_name))
	progressbar 	= tqdm(total = len(frac_range), desc = 'Frac loop (max: {0}) - working on {1}'.format(frac_range[-1], subject_name), disable = not enable_progressbar)
	
	try:
		# train and test for all fractions
		subject_dict ={'fracs': {'frac_'+str(frac): frac_model(estimator, d_val, d_test, skf_val, parameters, label_type, frac, progressbar) for frac in frac_range} }
		
	except:
		PrintException()
		sys.exit(1)
		
	# close the progressbar
	progressbar.close()
	
	return subject_dict
	
	
