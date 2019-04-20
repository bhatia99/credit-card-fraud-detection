import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('creditcard.csv')

print(dataset.info())
print('\n')
print(dataset.describe())

#gives a bird eye view of checking the missing values
#no missing values were found
sns.heatmap(dataset.isnull(),yticklabels=False,cmap='viridis',cbar=False)
plt.savefig('null_val_check.png')

"""Since the dataset is quite large with about 284k records,we can 
   take only a fraction of it through sampling to save computational resources.
   Although results maybe better if we use the entire dataset."""

#Code for sampling
dataset = dataset.sample(frac = 0.25, random_state = 1)
print(dataset.shape)

#exploring data
#plotting histogram of each feature
dataset.hist(figsize=(20,20))
plt.savefig('hist.png', bbox_inches='tight')
plt.show()

#we notice very few number of Fraudulent transactions
#no. of occurance of Legitimate/Valid(0) and Fraudulent(1) transactions
num_trans = dataset['Class'].value_counts()
valid = num_trans[0]
fraud = num_trans[1]
print('\nTotal number of tranactions : {}'.format(valid+fraud))
print('Number of Fraudulent tranactions : {}'.format(fraud))
print('Number of Valid tranactions : {}'.format(valid))

#calculating outlier fraction = fraudulent/total transactions
outlier_fraction = fraud/(fraud+valid)
print('Outlier Fraction : {}'.format(outlier_fraction))

#building a correlation matrix
corrmat = dataset.corr()
fig = plt.figure(figsize=(30,20))
sns.heatmap(corrmat)
plt.savefig('correlation.png')
plt.show()

#separating features and target
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

models = []
models.append(['LocalOutlierFactor',LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)])
models.append(['IsolationForest',IsolationForest(contamination = outlier_fraction, random_state = 1)])

for name,model in models:
    classifier = model
    if (name=='LocalOutlierFactor'):
        y_pred = classifier.fit_predict(X)
    else:
        classifier.fit(X)
        y_pred = classifier.predict(X)
    
    #predicted values will be 1 for inliners an -1 for outliers, we change them back to 0 and 1 respectively
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    print('\nFor {}:'.format(name))
    print('Accuracy: '+ str(accuracy_score(y,y_pred)))
    print('Confusion Matrix:')
    print(confusion_matrix(y,y_pred))
    print('Classification Report:')
    print(classification_report(y,y_pred))
        