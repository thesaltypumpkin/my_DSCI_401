import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

#from our class repo 
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))


##MAKE A BASE MODEL##	

df = pd.read_csv('./data/AmesHousingSetA.csv')

#give categorical features dummy variables 
df = pd.get_dummies(df, columns = cat_features(df))


#Set up a data X and data Y
del df['PID']
features = list(df)
features.remove('SalePrice')

data_x = df[features]
data_y = df['SalePrice']

#fix NaN in Data X
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x = imp.fit_transform(data_x) #imp.transform on data(x) 

#train/test split 
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)


#make base model
base_model = linear_model.LinearRegression()
base_model.fit(x_train,y_train)


base_model_preds = base_model.predict(x_test)

#print results 
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('Base Model R^2, EVS: ' + str([r2_score(y_test, base_model_preds), explained_variance_score(y_test, base_model_preds)])) 

##MODEL 2: BASE MODEL LASSO REGRESSION WITH AN ALPH OF 5.0 BEST MODEL##

lasso_mod_base = linear_model.Lasso(alpha=5, normalize=True, fit_intercept=True)
lasso_mod_base.fit(x_train, y_train)
preds = lasso_mod_base.predict(x_test)
print('Lasso Base Model R^2, EVS: ' + str([r2_score(y_test, preds), explained_variance_score(y_test, preds)])) 


##MODEL 3: WITHOUT CATEGORICAL FEATURES##

df_remove_cat = pd.read_csv('./data/AmesHousingSetA.csv')
cat_f = cat_features(df_remove_cat)

for i in cat_f: 
	del df_remove_cat[i] 

del df_remove_cat['PID']
features_rc = list(df_remove_cat)
features_rc.remove('SalePrice')

data_x_rc = df_remove_cat[features_rc]
data_y_rc = df_remove_cat['SalePrice']
	
#fix NaN
imp1 = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x_rc = imp1.fit_transform(data_x_rc) 

#train/test split 
x_train_rc, x_test_rc, y_train_rc, y_test_rc = train_test_split(data_x_rc, data_y_rc, test_size = 0.2, random_state = 4)

#make a model 
model_rc = linear_model.LinearRegression()
model_rc.fit(x_train_rc, y_train_rc)

model_rc_preds = model_rc.predict(x_test_rc)

print('categorical features removed R^2, EVS: ' + str([r2_score(y_test_rc, model_rc_preds), explained_variance_score(y_test_rc, model_rc_preds)])) 

##Model 4: Hand Picked features from numerical only##

#hand pick the features
#V.1 Features are not categorical in nature, picked after looking at graphs and noticing upward linear correlations
#V.2 added back in categorical features
#IMPORTANT NOTE: did feature selector methods with this model, and got consistently worse R^2 scores when compared to base model. 
df_1 = pd.read_csv('./data/AmesHousingSetA.csv')
df_hand_picked_one = pd.DataFrame()
df_hand_picked_one["Overall.Qual"]  = df_1["Overall.Qual"]   
df_hand_picked_one["Year.Built"] = df_1["Year.Built"]  
df_hand_picked_one["Year.Remod.Add"]  = df_1["Year.Remod.Add"]  
df_hand_picked_one["Bsmt.Unf.SF"]  = df_1["Bsmt.Unf.SF"]  
df_hand_picked_one["Total.Bsmt.SF"] = df_1["Total.Bsmt.SF"]  
df_hand_picked_one["X1st.Flr.SF"]  = df_1["X1st.Flr.SF"]  
df_hand_picked_one["Gr.Liv.Area"]  = df_1["Gr.Liv.Area"]  
df_hand_picked_one["Garage.Area"] = df_1["Garage.Area"]  
df_hand_picked_one["SalePrice"] = df_1["SalePrice"]  
df_hand_picked_one["Mas.Vnr.Area"] = df_1["Mas.Vnr.Area"]
df_hand_picked_one["LA"] = df_1["Lot.Area"]

#add back in categorical features 
cat_add = cat_features(df_1)

for i in cat_add: 
	df_hand_picked_one[i] = df_1[i]



df_hand_picked_one = pd.get_dummies(df_hand_picked_one, columns = cat_features(df_hand_picked_one))


features = list(df_hand_picked_one)
features.remove('SalePrice')

data_x_hp_1 = df_hand_picked_one[features]
data_y_hp_1 = df_hand_picked_one['SalePrice']



#fix NaN in Data X
imp3 = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x_hp_1 = imp3.fit_transform(data_x_hp_1)

#train/test split 
x_train_hp_1, x_test_hp_1, y_train_hp_1, y_test_hp_1 = train_test_split(data_x_hp_1, data_y_hp_1, test_size = 0.2, random_state = 4)

#make the model 
hand_picked_one_model = linear_model.LinearRegression()
hand_picked_one_model.fit(x_train_hp_1,y_train_hp_1)


hand_picked_one_preds = hand_picked_one_model.predict(x_test_hp_1)

#print results 
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('Hand Picked (and categorical) R^2, EVS: ' + str([r2_score(y_test_hp_1, hand_picked_one_preds), explained_variance_score(y_test_hp_1, hand_picked_one_preds)])) 

##MODEL 5: HAND PICKED MODEL WITH LASSO REGRESSION ALPHA = 7##

lasso_mod_hp = linear_model.Lasso(alpha=7, normalize=True, fit_intercept=True)
lasso_mod_hp.fit(x_train_hp_1, y_train_hp_1)
preds = lasso_mod_hp.predict(x_test_hp_1)
print('Hand Picked (and Categorical) Lasso R^2, EVS: ' + str([r2_score(y_test_hp_1, preds), explained_variance_score(y_test_hp_1, preds)])) 


#validate on base model 
#GOT AN ERROR HERE DISCUSED IN WRITE UP
#df_v_b = pd.read_csv('./data/AmesHousingSetB.csv')

#give categorical features dummy variables 
#df_v_b = pd.get_dummies(df_v_b, columns = cat_features(df_v_b))


#Set up a data X and data Y
#del df_v_b['PID']
#features = list(df_v_b)
#features.remove('SalePrice')

#data_x_v = df_v_b[features]
#data_y_v = df_v_b['SalePrice']

#fix NaN in Data X
#print("cant trasfomr")
#data_x_v = imp.fit_transform(data_x_v) #imp.transform on data(x) 

#base_model_v_preds = base_model.predict(data_x_v)

#print results 
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
#print('Base Model R^2, EVS: ' + str([r2_score(data_y_v, base_model_v_preds), explained_variance_score(data_y_v, base_model_v_preds)])) 