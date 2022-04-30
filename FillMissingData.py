import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris

#fill in missing data
iris=load_iris()
#print(iris)  #dict{'data':array[300:4],'target':array[150],'frame':None,'target_names':array[3],'DESCR':string}
x=iris.data
y=iris.target
print(y.size) #150

#using numpy.nan
masking_array=np.random.binomial(1,.25,x.shape).astype(bool)
x[masking_array]=np.nan                        #set as nan
print(masking_array.size)   #600
print(x.size)   #600
print(masking_array[:5])
print(x[:5])

#using SimpleImputer
impute=SimpleImputer(strategy='median')         #median
prime=impute.fit_transform(x)
print(prime[:5])

#using pandas.DataFrame
prime=np.where(pd.DataFrame(x).isnull(),-1,x)    #method1: set as -1
print(prime[:5])
prime=pd.DataFrame(x).fillna(-1)[:5].values      #method2: set as -1
print(prime)