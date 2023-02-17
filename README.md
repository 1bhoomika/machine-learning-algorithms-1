
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

from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
print('Missing values before imputation:', dataset.isnull().sum())
values = dataset.values
print('\nValues before imputation:\n', values)
imputer = SimpleImputer(missing_values=nan, strategy='median')
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
print('\nMissing values after imputation:', isnan(transformed_values).sum())



c) Mode

from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
print('Missing values before imputation:', dataset.isnull().sum())
values = dataset.values
print('\nValues before imputation:\n', values)
imputer = SimpleImputer(missing_values=nan, strategy='most_frequent')
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
print('\nMissing values after imputation:', isnan(transformed_values).sum())


d) Constant


from numpy import nan
from numpy import isnan
import pandas as pd
from sklearn.impute import SimpleImputer
dataset = pd.read_csv('pima-indians-diabetes.csv', header=None)
print('Missing values before imputation:', dataset.isnull().sum())
values = dataset.values
print('\nValues before imputation:\n', values)
imputer = SimpleImputer(missing_values=nan, strategy='constant')
transformed_values = imputer.fit_transform(values)
print('\nValues after imputation:\n', transformed_values)
print('\nMissing values after imputation:', isnan(transformed_values).sum())
