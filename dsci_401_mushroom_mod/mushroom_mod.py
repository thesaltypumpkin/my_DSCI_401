#Brian Will 
#Version 1.0.0
#Mushroom model
#Data from: https://www.kaggle.com/uciml/mushroom-classification
#Creates a model that will determine if a mushroom is edible or poison. 
#Builds an application with that model to take three inputs from a user and determine if a mushroom is edible or poison
#THIS MODEL IS FOR EDUCATIONAL PURPOSES. DO NOT USE IT IN A REAL LIFE SITUATION. IS NOT 100% ACCURATE.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from data_util import *
from sklearn.feature_extraction import DictVectorizer as DV

#CREATE BASE MIODEL#
def base_mod():
	data = pd.read_csv('./mushrooms.csv')
	data.drop_duplicates()

	#STEP ONE: PREPARE DATA#
	features = list(data)
	features.remove('class')
	data_x = data[features]
	data_y = data['class']
	le = preprocessing.LabelEncoder()
	le.fit(data_y)
	data_y = le.transform(data_y)
	data_x_dict = data_x.to_dict(orient = 'records')
	v = DV(sparse = False)
	data_x_dict = v.fit_transform(data_x_dict)
	


	#STEP TWO: SPLIT THE DATA#
	x_train, x_test, y_train, y_test = train_test_split(data_x_dict, data_y, test_size = 0.3)

	#STEP THREE: CREATE MODEL#
	print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
	dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
	dtree_gini_mod.fit(x_train, y_train)
	preds_gini = dtree_gini_mod.predict(x_test)
	print_multiclass_classif_error_report(y_test, preds_gini)


	#STEP FOUR: VALIDATE MODEL#

	print('----------- VALIDATE: DTREE WITH GINI IMPURITY CRITERION ------------------')
	data_v = pd.read_csv('./m_v.csv')

	features_v = list(data)
	features_v.remove('class')
	data_x_v = data_v[features]
	data_y_v = data_v['class']

	data_y_v = le.transform(data_y_v)
	data_x_dict_v = data_x_v.to_dict(orient = 'records')
	data_x_dict_v = v.transform(data_x_dict_v)

	preds_gini_v = dtree_gini_mod.predict(data_x_dict_v)
	print_multiclass_classif_error_report(data_y_v, preds_gini_v)

#CREATE A LIST OF MODELS USING EACH PREDICTOR AND ONLY THAT PREDICTOR#
def mod_list(): 
	data = pd.read_csv('./mushrooms.csv')
	accuracy_dict = {} 
	features = list(data)
	features.remove('class')
	for x in features: 
		data_x = data[x]
		data_y = data['class']
		
 		data_x = pd.get_dummies(data_x)
		le = preprocessing.LabelEncoder()
		le.fit(data_y)
		data_y = le.transform(data_y)
		
		x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3)


		##DTree Gini base model##
		print('----------- DTREE WITH GINI IMPURITY CRITERION:' + " " + x)
		dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
		dtree_gini_mod.fit(x_train, y_train)
		preds_gini = dtree_gini_mod.predict(x_test)
		print_multiclass_classif_error_report(y_test, preds_gini)
		
		accuracy_dict.update({x:accuracy_score(y_test, preds_gini)})
		
	print('---------ACCURACY DICTIONARY SORTED--------')
		
	for key, value in sorted(accuracy_dict.iteritems(), key=lambda (k,v): (v,k)):
		print "%s: %s" % (key, value)

#SIMPLE MOD V1#	
def simple_mod():
	data = pd.read_csv('./mushrooms.csv')
	data.drop_duplicates()

	#STEP ONE: PREPARE DATA#
	features = ['odor','spore-print-color', "gill-color" ]
	print(features)
	data_x =data[features]
	data_y = data['class']
	le = preprocessing.LabelEncoder()
	le.fit(data_y)
	data_y = le.transform(data_y)
	data_x_dict = data_x.to_dict(orient = 'records')
	v = DV(sparse = False)
	data_x_dict = v.fit_transform(data_x_dict)
	


	#STEP TWO: SPLIT THE DATA#
	x_train, x_test, y_train, y_test = train_test_split(data_x_dict, data_y, test_size = 0.3)

	#STEP THREE: CREATE MODEL#
	print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
	dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
	dtree_gini_mod.fit(x_train, y_train)
	preds_gini = dtree_gini_mod.predict(x_test)
	print_multiclass_classif_error_report(y_test, preds_gini)


	#STEP FOUR: VALIDATE MODEL#

	print('----------- VALIDATE: DTREE WITH GINI IMPURITY CRITERION ------------------')
	data_v = pd.read_csv('./m_v.csv')

	features_v = list(data)
	features_v.remove('class')
	data_x_v = data_v[features]
	data_y_v = data_v['class']

	data_y_v = le.transform(data_y_v)
	data_x_dict_v = data_x_v.to_dict(orient = 'records')
	data_x_dict_v = v.transform(data_x_dict_v)

	preds_gini_v = dtree_gini_mod.predict(data_x_dict_v)
	print_multiclass_classif_error_report(data_y_v, preds_gini_v)	

	return(dtree_gini_mod, le, v)
	
#SIMPLE MOD V2#
def simple_mod_v1():
	data = pd.read_csv('./mushrooms.csv')
	data.drop_duplicates()

	#STEP ONE: PREPARE DATA#
	features = ['stalk-color-above-ring','spore-print-color', "gill-color" ]
	print(features)
	data_x =data[features]
	data_y = data['class']
	le = preprocessing.LabelEncoder()
	le.fit(data_y)
	data_y = le.transform(data_y)
	data_x_dict = data_x.to_dict(orient = 'records')
	v = DV(sparse = False)
	data_x_dict = v.fit_transform(data_x_dict)
	


	#STEP TWO: SPLIT THE DATA#
	x_train, x_test, y_train, y_test = train_test_split(data_x_dict, data_y, test_size = 0.3)

	#STEP THREE: CREATE MODEL#
	print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
	dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
	dtree_gini_mod.fit(x_train, y_train)
	preds_gini = dtree_gini_mod.predict(x_test)
	print_multiclass_classif_error_report(y_test, preds_gini)


	#STEP FOUR: VALIDATE MODEL#

	print('----------- VALIDATE: DTREE WITH GINI IMPURITY CRITERION ------------------')
	data_v = pd.read_csv('./m_v.csv')

	features_v = list(data)
	features_v.remove('class')
	data_x_v = data_v[features]
	data_y_v = data_v['class']

	data_y_v = le.transform(data_y_v)
	data_x_dict_v = data_x_v.to_dict(orient = 'records')
	data_x_dict_v = v.transform(data_x_dict_v)

	preds_gini_v = dtree_gini_mod.predict(data_x_dict_v)
	print_multiclass_classif_error_report(data_y_v, preds_gini_v)	

	return(dtree_gini_mod, le, v)

#RUDEMENTRY APPLICATION USING THE SIMPLE MODS#	
def app(): 
	
	model = raw_input("Can you smell the mushroom? (y/n) ")
	print("")
	
	if model =='y':
		mod, le, v = simple_mod()
		print("Odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s, unknown = u")
		scent = raw_input("what is the mushrooms scent? ") 
		print("")
		print("Spore print color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y, other = ot")
		spc = raw_input("what is the mushrooms spore print color?") 
		print("")
		print("gill color: black=k, brown=n, buff=b, chocolate=h, gray=g, other = ot")
		gc = raw_input("what is the mushrooms gill color?") 
		print("")
	
		mushroom = {'odor': scent, 'spore-print-color':spc, 'gill-color':gc} 
		mushroom = v.transform(mushroom)
		pred = mod.predict(mushroom)
		pred = le.inverse_transform(pred) 
	
		if pred[0] == 'e': 
			print("Chances are the mushroom is edible")
		if pred[0] == 'p': 
			print("You'll probably die if you eat that, don't risk it")

	else: 
		mod, le, v = simple_mod_v1()
		print("Stalk color above the ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y, unknown=u")
		scar = raw_input("what is the stalk color above the ring? ") 
		print("")
		print("Spore print color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y, other = ot")
		spc = raw_input("what is the mushrooms spore print color? ") 
		print("")
		print("gill color: black=k, brown=n, buff=b, chocolate=h, gray=g, other = ot")
		gc = raw_input("what is the mushrooms gill color? ") 
		print("")
	
		mushroom = {'stalk-color-above-ring': scar, 'spore-print-color':spc, 'gill-color':gc} 
		mushroom = v.transform(mushroom)
		pred = mod.predict(mushroom)
		pred = le.inverse_transform(pred) 
	
		if pred[0] == 'e': 
			print("Chances are the mushroom is edible")
		if pred[0] == 'p': 
			print("You'll probably die if you eat that, don't risk it")


app()	