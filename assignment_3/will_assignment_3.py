import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import preprocessing
from data_util import *

data = pd.read_csv('./data/churn_data.csv')

def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

data = pd.get_dummies(data, columns = ['Gender', 'Income'])

##Create decision tree Base Models##

# Select x and y data
del data["CustID"]
features = list(data)
features.remove('Churn')
features.remove('Gender_Male')
features.remove('Income_Upper')
data_x = data[features]
data_y = data['Churn']

# Convert the different class labels to unique numbers with label encoding.
le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

##DTree Gini base model##
print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(x_train, y_train)
preds_gini = dtree_gini_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini)

##DTree Entropy base model##
print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)

##Gini base model did better... so we will modify that..##
print('----------- DTREE WITH GINI IMPURITY CRITERION, MODIFIED ------------------')
dtree_gini_v1 = tree.DecisionTreeClassifier(criterion='gini', max_depth = 6)
dtree_gini_v1.fit(x_train, y_train)
preds_gini_v1 = dtree_gini_v1.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_gini_v1)

#Normalize data#
data_x_norm = preprocessing.normalize(data_x, axis=0)
x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(data_x_norm, data_y, test_size = 0.3, random_state = 4)

##Grid search k-nearest Neighbor##
#ks = [2, 3, 6, 8, 10, 12, 14, 16, 18, 20]
#for k in ks:
#	# Create model and fit.
#	mod = neighbors.KNeighborsClassifier(n_neighbors=k)
#	mod.fit(x_train_n, y_train_n)

	# Make predictions - both class labels and predicted probabilities.
#	preds = mod.predict(x_test_n)
#	print('---------- EVALUATING MODEL: k = ' + str(k) + ' -------------------')
#	print_multiclass_classif_error_report(y_test_n, preds)

##k-nearest neighbors grid search resulted in 2... so here it is##
knn_mod = neighbors.KNeighborsClassifier(n_neighbors=2)
knn_mod.fit(x_train_n, y_train_n)
preds = knn_mod.predict(x_test_n)
print('---------- EVALUATING MODEL: k = 2 -------------------')
print_multiclass_classif_error_report(y_test_n, preds)

##naive bays base model... ##
print('----------- NAIVE BAYES GAUSSIAN ------------------')
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)
##actually turns out to be pretty good... might be worth trying to modify...##

#validate base models#

vl = pd.read_csv('./data/churn_validation.csv')

vl = pd.get_dummies(vl, columns = ['Gender', 'Income'])
# Select x and y data
del vl["CustID"]
features_vl = list(vl)
features_vl.remove('Churn')
features_vl.remove('Gender_Male')
features_vl.remove('Income_Upper')
data_x_v = vl[features]
data_y_v = vl['Churn']

data_y_v = le.transform(data_y_v)

print('\n----------- Validate_Trees -----------------------')
print('----------- DTREE WITH GINI IMPURITY CRITERION ------------------')
pred_v = dtree_gini_mod.predict(data_x_v)
print_multiclass_classif_error_report(data_y_v, pred_v)

print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
preds_v_2 = dtree_entropy_mod.predict(data_x_v)
print_multiclass_classif_error_report(data_y_v, preds_v_2)

print('----------- NAIVE BAYES GAUSSIAN ------------------')
preds_v_3 = gnb_mod.predict(data_x_v)
print_multiclass_classif_error_report(data_y_v, preds_v_3)

print('----------- DTREE WITH GINI IMPURITY CRITERION, MODIFIED ------------------')
preds_v_4 = dtree_gini_v1.predict(data_x_v)
print_multiclass_classif_error_report(data_y_v, preds_v_4)

print('---------- EVALUATING MODEL: k = 2 -------------------')
data_x_v_n = preprocessing.normalize(data_x_v, axis=0)
preds_v_5 = knn_mod.predict(data_x_v_n)
print_multiclass_classif_error_report(data_y_v, preds_v_5)


