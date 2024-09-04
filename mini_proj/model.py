import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data =pd.read_csv("heart study.csv")

framingham=data.dropna()

#'duplicated()' function in pandas return the duplicate row as True and othter as False
#for counting the duplicate elements we sum all the rows
sum(data.duplicated())
data = data[data['totChol']<696.0]
data = data[data['sysBP']<295.0]
data.shape


#dropping the education column because its not related
df = data.drop(['education'], axis=1)
X=df.drop(['TenYearCHD'], axis=1)
y=df['TenYearCHD']
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# replace missing values with the median (axis=0 means column-wise)
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# apply Select Best class to extract top 10 best featuresa
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concatenate two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 

#print(featureScores.nlargest(11,'Score'))  # print 10 best features
featureScores = featureScores.sort_values(by='Score', ascending=False)

# selecting the 10 most impactful features for the target variable
features_list = featureScores["Specs"].tolist()[:10]
print(features_list)
# Create new dataframe with selected features

df = df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]
X=X[['male','age','cigsPerDay','BPMeds','prevalentHyp','diabetes','totChol','sysBP','diaBP','glucose']]
from sklearn.model_selection import train_test_split

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# find the length of the training and testing datasets
print("Length of training dataset:", len(X_train))
print("Length of testing dataset:", len(X_test))

'''from sklearn.svm import SVC


# Train the support vector machine classifier
model = SVC()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print(y_pred)
print(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)'''


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
model= KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred)
print(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

import flask as fs 
print('s')

import pickle

pickle.dump(model,open("model.pkl","wb"))
