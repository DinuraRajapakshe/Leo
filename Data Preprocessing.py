import matplotlib.pyplot as plt
import pandas as pd

data_set1 = pd.read_csv ('Data_set.csv')
data_set2 = pd.read_csv ('Data_Set.csv', header = 2)
data_set3 = data_set2.rename (columns = {'Temperature': 'Temp'})
data_set4 = data_set3.drop ('No. Occupants', axis = 1)

# without making new variables we can use inplace tool. Inplace = True implicate the operations
# data_set3.drop('No. Occupants', axis= 1,inplace= True)

# dropping a row
data_set4.drop (2, axis = 0, inplace = True)

# re-indexing the columns
data_set6 = data_set4.reset_index (drop = True)

# describe the data [mean, var, SD.....]
data_set6.describe ()  # info and statistics of the dataframe

# Locating a specific value in a dataset
min_item = data_set6 ['E_Heat'].min ()
# data_set6['E_Heat'][data_set6['E_Heat'] == min_item]  ?recheck

# replacing an unwanted value in a dataset
data_set6 ['E_Heat'].replace (-4, 21, inplace = True)

# Covariance

data_set6.cov ()  # = Variable name

# visualising data set on a heat map. correlation
import seaborn as sn

sn.heatmap (data_set6.corr ())
plt.show ()

# Missing Values

# data_set6.info ()

import numpy as np

data_set7 = data_set6.replace ('!', np.NaN)

# data_set7.info ()

# changing the nature of the "price" column to numeric
data_set7 = data_set7.apply (pd.to_numeric)

# data_set7.info ()

# Finding_Nullvalues = data_set7.isnull()
# dropping a single row
# data_set7.drop(13, axis = 0,inplace = True)
#
# # dropping multiple rows which contains NaN
# data_set7.dropna( axis = 0, inplace = True)

# Fill the NaN cells with values

data_set8 = data_set7.fillna (method = 'ffill')  # 'ffill'  = we're using previous value to fill in the NaN cell
# 'bfill' =  we're using the following value to  fill in the NaN cell

# filling the missing values with median or more frequent data
from sklearn.impute import SimpleImputer  # importing a library from Scikitlearn

M_var = SimpleImputer (missing_values = np.NaN, strategy = 'mean')

# fitting the Imputer is important.
M_var.fit (data_set7)

data_set9 = M_var.transform (data_set7)

# Outlier
# Mild Outliers
#          Q1 - 1.5 x IQR(Lower)
#          Q3 + 1.5 x IQR(Upper)

# Extreme Outliers
#         Q1 - 3 x IQR(Lower)
#         Q@ + 3 x IQR(Upper)

# Outlier Detection

data_set8.boxplot ()
plt.show ()

Q1 = data_set8 ['E_Plug'].quantile(0.25)
Q2 = data_set8 ['E_Plug'].quantile(0.75)

# Q1 = 21.25
# Q3 = 33.75
# IQR = Q3-Q1 = 12.5

# Mild Outlier
# Lower = Q1 - 1.5 X IQR = 21.25 - 1.5 x 12.5 = 2.5
# upper = Q3 + 1.5 x IQR = 33.75 + 1.5 x 12.5 = 52.5

# Extreme Outlier
# Lower = Q1 - 3 X IQR = -16.25
# Upper = Q3 + 3 x IQR =  71.25

# Hence 120 in "E_Plug" is considered to be an outlier.

# Removing the Outlier
data_set8['E_Plug'].replace(120, 42, inplace = True)

                        # Preprocessing Concatenating

new_column = pd.read_csv('Data_New.csv')

# Attached the column to the dataframe
data_set10 = pd.concat([data_set8,new_column],axis = 1)

                            # Dummy Variable

# data_set10.info()

data_set11 = pd.get_dummies(data_set10)

# data_set11.info()

                            # Preprocessing Normalization
# bringing the data set into a same range
from sklearn.preprocessing import minmax_scale, normalize

# First Method : Min Max scale

data_set12 = minmax_scale(data_set11,feature_range = (0,1))

data_set13 = normalize(data_set11,norm = 'l2',axis = 0) # l2 = euclinian method
# axis = 0 is for normalize each feature. axis = 1 is for normalizing each sample.
 # l1 = manhatton method

# coverting datasets into dataframe format
data_set13 = pd.DataFrame(data_set13, columns = ['Time','E_plug','E_Heat','Price',
                                                 'Temp','OffPeak','Peak'])

data_set12 = pd.DataFrame(data_set12, columns = ['Time','E_plug','E_Heat','Price',
                                                 'Temp','OffPeak','Peak'])



