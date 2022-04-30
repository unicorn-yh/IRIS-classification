from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

iris=load_iris()
x=iris.data
X=iris.data[:,:2]  #load the iris data #numpy.ndarray
y=iris.target   
print(type(X))
print(type(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=1)
clf=SVC(kernel='linear') #train the model
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test) #sets of predicted values
score=accuracy_score(y_test,y_pred)  #examine the model acccuracy 
print(score)

'''print("Multi-Output Scores for the Iris Flowers: ")
for column in range(0,3):
    print("Accracy score of flower"+str(column),accuracy_score(y_test[:,column],y_pred[:,column]))
    print("AUC score of flower"+str(column),roc_auc_score(y_test[:,column],y_pred[:,column]))
    print("")'''



'''print("x="+str(x.size)) #600
print("X="+str(x.size)) #600
print("y="+str(y.size)) #150
print("X_train="+str(X_train.size)) #360
print("X_test="+str(X_test.size))   #90
print("y_train="+str(y_train.size)) #120
print("y_test="+str(y_test.size))   #30
print("y_pred="+str(y_pred.size))   #30'''

#LinearSVC
'''from sklearn.svm import LinearSVC
clf=LinearSVC(random_state=1)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))'''



#regression
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=1)  #train the model
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test) #predict with logistic regression
print(accuracy_score(y_test,y_pred))