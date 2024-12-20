import pandas as pd
import numpy as np
import warnings

df=pd.read_csv(r"C:\Users\Likhith\OneDrive\Desktop\breastcancer.csv")
print(df)
df.info()

# data cleaning

df.drop(df.columns[[-1,0]],axis = 1, inplace = True)
print(df)
df['diagnosis'].value_counts()
print(df)

# feature selection

from sklearn.model_selection import train_test_split
diag_map = {'M' : 1, 'B' : 0}
df['diagnosis'] = df['diagnosis'].map(diag_map)
print(df)

x = df[['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']]
print(x)

y = df[['diagnosis']]
x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=42)

#model knn

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_y_pred = knn.predict(x_test)
print(knn)

from sklearn.metrics import accuracy_score
print(accuracy_score(knn_y_pred, y_test))

#model - Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
lr_y_pred = lr.predict(x_test)
print(lr)
print(accuracy_score(lr_y_pred, y_test))

#model - naive bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_y_pred = gnb.predict(x_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, gnb_y_pred))

#model - K cross validation

from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

accuracy_all=[]
cvs_all=[]

#knn
scores = cross_val_score(knn, x, y, cv = 10)
accuracy_all.append(accuracy_score(knn_y_pred, y_test))
cvs_all.append(np.mean(scores))

print("knn Accuracy: {0:.2%}".format(accuracy_score(knn_y_pred, y_test)))
print("cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores),np.std(scores)*2))

# logistic regression

accuracy_all=[]
cvs_all=[]
scores = cross_val_score(lr, x, y, cv = 10)
accuracy_all.append(accuracy_score(lr_y_pred, y_test))
cvs_all.append(np.mean(scores))

print("logistic regression Accuracy: {0:.2%}".format(accuracy_score(lr_y_pred, y_test)))
print("cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores),np.std(scores)*2))

#naive bayes

accuracy_all=[]
cvs_all=[]
scores = cross_val_score(gnb, x, y, cv = 10)
accuracy_all.append(accuracy_score(gnb_y_pred, y_test))
cvs_all.append(np.mean(scores))

print("Naive Bayes Accuracy: {0:.2%}".format(accuracy_score(gnb_y_pred, y_test)))
print("cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores),np.std(scores)*2))

 



