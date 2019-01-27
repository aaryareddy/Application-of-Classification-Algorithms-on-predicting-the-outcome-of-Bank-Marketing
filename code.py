# Classification algorithms
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graphviz as gv

# Importing the dataset
dataset = pd.read_csv('bank-additional-full-d.csv')

#removing duation attribute
dataset = dataset.drop(['duration'], axis=1)
# Removing missing values
dataset = dataset.replace("unknown", np.NaN)
dataset.dropna(inplace=True)

# One hot encoding
print('Original Features:\n', list(dataset.columns), '\n')
data_dummies = pd.get_dummies(dataset)
print('Features after One-Hot Encoding:\n', list(data_dummies.columns))

bankds = data_dummies.loc[:, 'age':'poutcome_success']
X = bankds.values
y = data_dummies['y_yes'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set------------------------------------------
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

classifier_lr100 = LogisticRegression(C=100, random_state = 0)
classifier_lr100.fit(X_train, y_train)

classifier_lr001 = LogisticRegression(C=0.01, random_state = 0)
classifier_lr001.fit(X_train, y_train)

# Predicting the Test set results
y_pred_lr = classifier_lr.predict(X_test)
print(y_pred_lr)

y_pred_lr100 = classifier_lr100.predict(X_test)
print(y_pred_lr100)

y_pred_lr001 = classifier_lr001.predict(X_test)
print(y_pred_lr001)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

cm_lr100 = confusion_matrix(y_test, y_pred_lr100)
print(cm_lr100)

cm_lr001 = confusion_matrix(y_test, y_pred_lr001)
print(cm_lr001)


print('Accuracy on the training subset {:.3f}'.format(classifier_lr.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_lr.score(X_test, y_test)))

print('Accuracy on the training subset: {:.3f}'.format(classifier_lr100.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_lr100.score(X_test, y_test)))

print('Accuracy on the training subset: {:.3f}'.format(classifier_lr001.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_lr001.score(X_test, y_test)))

plt.plot(classifier_lr.coef_.T, 'o', label='C=1')
plt.plot(classifier_lr100.coef_.T, '^', label='C=100')
plt.plot(classifier_lr001.coef_.T, 'v', label='C=0.01')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
plt.hlines(0,0, bankds.data.shape[1])
plt.xticks(56,bankds.feature_names, rotation=90)
plt.legend()

# Fitting K-NN to the Training set---------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)


# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)
print(y_pred_knn)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

print('Accuracy on the training subset {:.3f}'.format(classifier_knn.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_knn.score(X_test, y_test)))



# Create two lists for training and test accuracies
training_accuracy = []
test_accuracy = []

# Define a range of 1 to 10 (included) neighbors to be tested
neighbors_settings = range(1,11)

# Loop with the KNN through the different number of neighbors to determine the most appropriate (best)
for n_neighbors in neighbors_settings:
   classifier_knn1 = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'minkowski', p = 2)
   classifier_knn1.fit(X_train, y_train)
   training_accuracy.append(classifier_knn1.score(X_train, y_train))
   test_accuracy.append(classifier_knn1.score(X_test, y_test))

# Visualize results - to help with deciding which n_neigbors yields the best results (n_neighbors=6, in this case)
plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()


# Fitting Naive Bayes to the Training set-----------------------------------------------------

from sklearn.naive_bayes import GaussianNB

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)


# Predicting the Test set results
y_pred_nb = classifier_nb.predict(X_test)
print(y_pred_nb)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb= confusion_matrix(y_test, y_pred_nb)
print(cm_nb)

print('Accuracy on the training subset {:.3f}'.format(classifier_nb.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_nb.score(X_test, y_test)))

# Fitting Decision Tree Classification to the Training set------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)


# Predicting the Test set results
y_pred_dt = classifier_dt.predict(X_test)
print(y_pred_dt)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

print('Accuracy on the training subset: {:.3f}'.format(classifier_dt.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_dt.score(X_test, y_test)))

# Fitting Decision Tree Classification to the Training set with max depth
classifier_dt_md = DecisionTreeClassifier(max_depth=4, random_state=0)
classifier_dt_md.fit(X_train, y_train)

# Predicting the Test set results for DT with max depth
y_pred_dt_md = classifier_dt_md.predict(X_test)
print(y_pred_dt_md)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dt_md = confusion_matrix(y_test, y_pred_dt_md)
print(cm_dt_md)

print('Accuracy on the training subset: {:.3f}'.format(classifier_dt_md.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_dt_md.score(X_test, y_test)))

#graph
from sklearn.tree import export_graphviz
dot_data = export_graphviz(classifier_dt_md, out_file= 'decisiontreefinal.dot', class_names=['Subscribed','Unsubscribed'],
                       impurity=False, filled=True)
graph = gv.Source(dot_data)  
print(graph)

# Fitting Random Forest Classification to the Training set---------------------------------

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test)
print(y_pred_rf)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

print('Accuracy on the training subset: {:.3f}'.format(classifier_rf.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(classifier_rf.score(X_test, y_test)))

#classifier features importance
classifier_rf.feature_importances_
  
# Fitting SVM to the Training set--------------------------------------------------------------

#from sklearn.svm import SVC
#classifier_svm = SVC(kernel = 'linear', random_state = 0)
#classifier_svm.fit(X_train, y_train)

# Predicting the Test set results
#y_pred_svm = classifier_svm.predict(X_test)
#print(y_pred_svm)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm_svm = confusion_matrix(y_test, y_pred_svm)
#print(cm_svm)

#print('The accuracy on the training subset: {:.3f}'.format(classifier_svm.score(X_train, y_train)))
#print('The accuracy on the test subset: {:.3f}'.format(classifier_svm.score(X_test, y_test)))

#scaled svm
#min_train = X_train.min(axis=0)
#range_train = (X_train - min_train).max(axis=0)

#X_train_scaled = (X_train - min_train)/range_train

#print('Minimum per feature\n{}'.format(X_train_scaled.min(axis=0)))
#print('Maximum per feature\n{}'.format(X_train_scaled.max(axis=0)))

#X_test_scaled = (X_test - min_train)/range_train

#classifier_svm1 = SVC(kernel = 'linear', random_state = 0)
#classifier_svm1.fit(X_train_scaled, y_train)

# Predicting the Test set results
#y_pred_svm1 = classifier_svm1.predict(X_test_scaled)
#print(y_pred_svm1)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm_svm1 = confusion_matrix(y_test, y_pred_svm1)
#print(cm_svm1)
#print('The accuracy on the training subset: {:.3f}'.format(classifier_svm1.score(X_train_scaled, y_train)))
#print('The accuracy on the test subset: {:.3f}'.format(classifier_svm1.score(X_test_scaled, y_test)))

#linear kernel svm with c=5
#classifier_svm2 = SVC(C=2, kernel = 'linear', random_state = 0)
#classifier_svm2.fit(X_train, y_train)

# Predicting the Test set results
#y_pred_svm2 = classifier_svm2.predict(X_test)
#print(y_pred_svm2)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm_svm2 = confusion_matrix(y_test, y_pred_svm2)
#print(cm_svm2)

#print('The accuracy on the training subset: {:.3f}'.format(classifier_svm2.score(X_train, y_train)))
#print('The accuracy on the test subset: {:.3f}'.format(classifier_svm2.score(X_test, y_test)))

# Fitting rbf Kernel SVM to the Training set---------------------------------------------------------

#from sklearn.svm import SVC
#classifier_ksvm = SVC(kernel = 'rbf', random_state = 0)
#classifier_ksvm.fit(X_train, y_train)

# Predicting the Test set results
#y_pred_ksvm = classifier_ksvm.predict(X_test)
#print(y_pred_ksvm)
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm_ksvm = confusion_matrix(y_test, y_pred_ksvm)
#print(cm_ksvm)

#print('The accuracy on the training subset: {:.3f}'.format(classifier_ksvm.score(X_train, y_train)))
#print('The accuracy on the test subset: {:.3f}'.format(classifier_ksvm.score(X_test, y_test)))
