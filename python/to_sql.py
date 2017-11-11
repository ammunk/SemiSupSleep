import json, pandas as pd
from sqlalchemy import Text, create_engine, Integer, String, Column, DateTime, Float, ForeignKey, Boolean, and_
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sqlalchemy, os
from sqlalchemy.types import TypeDecorator
import logging, json
from os import walk


SIZE = 256


class TextPickleType(TypeDecorator):

	"""
	
	You can create a custom type by subclassing sqlalchemy.types.TypeDecorator to handle serialization and deserialization to Text:
	 - https://stackoverflow.com/questions/1378325/python-dicts-in-sqlalchemy
	 - http://docs.sqlalchemy.org/en/latest/core/custom_types.html
	
	"""

    	impl = sqlalchemy.Text(SIZE)

    	def process_bind_param(self, value, dialect):
        	if value is not None:
            		value = json.dumps(value)

        	return value

    	def process_result_value(self, value, dialect):
        	if value is not None:
         	   value = json.loads(value)
        	return value

# echo controls how verbose sqlalchemy is - set to True to more information shown in the console
engine	 = create_engine('sqlite:///DataBase/sleep_performance.db', echo = False)

Base	 = declarative_base()


# we do a one to many relational database scheme

class Time(Base):
	__tablename__ 	= 'times'

	id 			= Column(Integer, primary_key = True)
	created_at 		= Column(DateTime, default = datetime.now(), onupdate = datetime.now())
	n_split			= Column(Integer)
	program_duration	= Column(String)
	cov_type		= Column(String)
	label_type		= Column(String)
	unlabel_weight		= Column(Boolean)
	unlabel_independent 	= Column(Boolean)
	subjects		= relationship("Subject")
	
	def __repr__(self):
        	return "<Time(time_id = {5}, created_at= {0}, n_split= {1}, program_duration= {2}, cov_type={3}, label_type={4})>".format(
                                    self.created_at, self.n_split, self.program_duration, self.cov_type, self.label_type, self.id)
	
class Subject(Base):
	__tablename__	= 'subjects'
	
	id		= Column(Integer, primary_key = True)
	subject		= Column(String)
	class_prior 	= Column(TextPickleType())	
	total_sleep 	= Column(Float)
	nnmf_dim	= Column(Integer)
	nnmf_error	= Column(Float)
	time_id		= Column(Integer, ForeignKey('times.id'))

	fracs		= relationship("Frac")

	def __repr__(self):
        	return "<Subject(subject_id = {6}, subject= {0}, class_prior= {1}, total_sleep= {2}, nnmf_dim={3}, nnmf_error={4}, time_id={5})>".format(
                                    self.subject, self.class_prior, self.total_sleep, self.nnmf_dim, self.nnmf_error, self.time_id, self.id)

class Frac(Base):
	__tablename__	= 'fracs'
	
	id			= Column(Integer, primary_key = True)
	frac			= Column(String)
	best_params 		= Column(TextPickleType())	
	best_val_score 		= Column(Float)	
	test_score		= Column(Float)
	means			= Column(TextPickleType())
	covariances		= Column(TextPickleType())
	confusion_matrix	= Column(TextPickleType())
	subject_id		= Column(Integer, ForeignKey('subjects.id'))
	
	def __repr__(self):
        	return "<Frac(frac_id = {8}, frac= {0}, best_params= {1}, best_val_score= {2}, test_score={3}, means = {4}, covariances = {5}, confusion matrix= {6}, subject_id={7})>".format(
                                    self.frac, self.best_params, self.best_val_score, self.test_score, self.means, self.covariances, self.confusion_matrix, self.subject_id, self.id)
    
### Create_all does not recreate/modify already existing tables. Thus if new columns needs added you have to drop old tables
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


# other scores are f1 and others.... Implement


Session		= sessionmaker(bind = engine)
session 	= Session()


# function for parsing all values into strings of the dict in order for the sql database for properly handle the dict
def parse_dict_to_string(params):
	return {key: str(value) for key, value in params.iteritems()}

def convert_synth_name(dict_name):
	if dict_name == 'SYNTH':
		return 'subject'
	else:
		return dict_name

def extract_non_dict(a_dict, dict_name = None):
	new_dict 		= None
	if dict_name:
		dict_name, name_num 	= dict_name.split("_")
		dict_name		= convert_synth_name(dict_name)
		extracted 		= {dict_name: dict_name + name_num}
	else:
		extracted 		= {}
	for key, value in a_dict.iteritems():
		
		if not isinstance(value,dict) :
			extracted.update({key:value})
		else:
			new_dict = value
	
	return extracted, new_dict

def to_sql(performance_dict):
	time_dict, run_results_dict = extract_non_dict(performance_dict)
	entry			= Time(**time_dict)

	for subject_key, sub_dict in run_results_dict.iteritems():
		extract_subject_info, frac_dict 		= extract_non_dict(sub_dict, subject_key)
		sub = Subject(**extract_subject_info)
		for frac_key, frac_dict in frac_dict.iteritems():
			extract_frac_info, frac_sql	= extract_non_dict(frac_dict, frac_key)
			sub.fracs.append(Frac(**extract_frac_info))
		
		entry.subjects.append(sub)
	
	session.add(entry)
	session.commit()
	
if __name__ == "__main__":
	current_dir 		= os.path.dirname(__file__)
	reselt_path 		= os.path.abspath(os.path.join(current_dir, "results"))

	for (dirpath, dirnames, filenames) in walk(reselt_path):
		for a_file in filenames:
			file_to_load = os.path.abspath(os.path.join(reselt_path, a_file))
			with open(file_to_load, 'r') as infile:
				result_dict = json.load(infile)
			to_sql(result_dict)
		# break the first time, since we only want to deal with the result folder and not subfolders
		break

