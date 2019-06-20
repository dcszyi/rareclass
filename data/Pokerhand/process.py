
import numpy as np 
import pandas as pd


df = pd.read_csv('poker-hand-training-true.data', sep=",", 
                    names=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "CLASS"])
print(df[:10])

df2 = pd.get_dummies(df, columns=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5"]) 

print(df2[:10])
#df.col3 = pd.Categorical.from_array(dataframe.col3).codes

values = df2.values
y = values[:,0]
X = values[:,1:]
print(X.shape, y.shape) 

np.save('./X.onehot.npy', X)
np.save('./y.multiclass.npy', y)

#Test
X = np.load('./X.onehot.npy')
y = np.load('./y.multiclass.npy')
#print(X[:10])
print(y[:10])


y_unique, y_counts = np.unique(y, return_counts=True)
print(dict(zip(y_unique, y_counts)))#print out basic statisitics of the dataset

for i,target_class in enumerate(y_unique):      
    y_binary =(y==target_class).astype(int)
    print(y_binary[:10])
    np.save('./y.binary_class'+str(target_class)+'.npy', y_binary)

 

