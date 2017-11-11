import pandas as pd
import numpy as np

# entries into enums are themselves instances, which have access to class methods
from enum import Enum, unique

def to_numpy(d_val, d_test, label_type):

	# make labels into 1d array-like	
	d_val	= (d_val['X'].as_matrix(), d_val[label_type].as_matrix().reshape(-1))
	d_test	= (d_test['X'].as_matrix(), d_test[label_type].as_matrix().reshape(-1))
	return d_val, d_test
	
def remove_nan(d_val, d_test):
	return d_val.dropna(axis=0, how = 'any'), d_test.dropna(axis=0, how = 'any')		

@unique
class y_c4(Enum):
	DEEP_SLEEP	= 1
	LIGHT_SLEEP	= 2
	REM		= 3
	WAKE		= 4

	@classmethod
	def sleep_types(cls):
		return [sleep_type.name for sleep_type in cls]

@unique
class y_c6(Enum):
	N4		= 1
	N3		= 2
	N2		= 3
	N1		= 4
	REM		= 5
	WAKE		= 6
	
	@classmethod
	def sleep_types(cls):
		return [sleep_type.name for sleep_type in cls]
	
		
@unique
class SYNTH(Enum):
	LAB1		= 1
	LAB2		= 2
	
	@classmethod
	def sleep_types(cls):
		return [sleep_type.name for sleep_type in cls]
		
@unique
class SYNTH_3(Enum):
	LAB1		= 1
	LAB2		= 2
	LAB3		= 3
	
	@classmethod
	def sleep_types(cls):
		return [sleep_type.name for sleep_type in cls]
