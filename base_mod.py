#Used to try to debug the validation process##

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

df_i = df 
cat = cat_features(df)
for i in cat: 
	del df_i[i]
	

#give categorical features dummy variables 
df = pd.get_dummies(df, columns = cat_features(df))


#Set up a data X and data Y
del df['PID']
features = list(df)
print(len(features))
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
#print('Base Model R^2, EVS: ' + str([r2_score(y_test, base_model_preds), explained_variance_score(y_test, base_model_preds)])) 

#validate on base model 
df_v_b = pd.read_csv('./data/AmesHousingSetB.csv')

#give categorical features dummy variables 
df_v_b = pd.get_dummies(df_v_b, columns = cat_features(df_v_b))


#Set up a data X and data Y
del df_v_b['PID']
features = list(df_v_b)
print(len(features))
features.remove('SalePrice')

data_x_v = df_v_b[features]
data_y_v = df_v_b['SalePrice']

#fix NaN in Data X
print("cant trasfomr")
data_x_v = imp.transform(data_x_v) #imp.transform on data(x) 

base_model_v_preds = base_model.predict(data_x_v)

#print results 
#pprint.pprint(pd.DataFrame({'Actual':y_test, 'Predicted':preds}))
print('Base Model R^2, EVS: ' + str([r2_score(data_y_v, base_model_v_preds), explained_variance_score(data_y_v, base_model_v_preds)])) 