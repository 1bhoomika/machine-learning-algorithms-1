# machine-learning-algorithms-1
This repository includes classification and regression algorithms.
# program to implement different data imputations in Machine Learning using Python.


a) Mean


from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
print('Missing values before imputation:', dataset.isnull().sum())
values = dataset.values
print('\nValues before imputation:\n', values)
imputer = SimpleImputer(missing_values=nan, strategy='mean')
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
print('\nMissing values after imputation:', isnan(transformed_values).sum())


b) Median

# import necessary libraries
from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
# load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
# Getting the count of missing values in each column 
print('Missing values before imputation:', dataset.isnull().sum())
# retrieve the numpy array
values = dataset.values
print('\nValues before imputation:\n', values)
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='median')
# transform the dataset
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
# Getting the count of missing values after imputation
print('\nMissing values after imputation:', isnan(transformed_values).sum())



c) Mode

# import necessary libraries
from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
# load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
# Getting the count of missing values in each column 
print('Missing values before imputation:', dataset.isnull().sum())
# retrieve the numpy array
values = dataset.values
print('\nValues before imputation:\n', values)
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='most_frequent')
# transform the dataset
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
# Getting the count of missing values after imputation
print('\nMissing values after imputation:', isnan(transformed_values).sum())


d) Constant

# import necessary libraries
from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
# load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
# Getting the count of missing values in each column 
print('Missing values before imputation:', dataset.isnull().sum())
# retrieve the numpy array
values = dataset.values
print('\nValues before imputation:\n', values)
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='constant')
# transform the dataset
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
# Getting the count of missing values after imputation
print('\nMissing values after imputation:', isnan(transformed_values).sum())
