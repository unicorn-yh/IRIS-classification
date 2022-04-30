from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class K_Means():
    def __init__(self):
        self.getData()
        self.cluster()
        self.visualize()
        self.elbowMethod()
        self.silhouetteCoefficient()

    def getData(self):
        iris=load_iris()
        self.X=iris.data[:,:2]
        self.y=iris.target
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.3,random_state=1)

    def data_preprocessing(self):
        pca=PCA()
        self.X_train=pca.fit_transform(self.X_train)
        self.X_test=pca.transform(self.X_test)
        scaler=StandardScaler()
        self.X_train_scaled=scaler.fit_transform(self.X_train.astype(np.float32))
        self.X_test_scaled=scaler.transform(self.X_test.astype(np.float32))

    def cluster(self):
        self.km = KMeans(n_clusters=len(np.unique(self.y)),init='k-means++',max_iter=100,n_init=10,random_state=0)
        km=self.km
        self.y_kmeans=km.fit_predict(self.X)
        km.fit(self.X_train,self.y_train)
        train_label=km.predict(self.X_train)
        test_label=km.predict(self.X_test)
        train_score=accuracy_score(self.y_train,train_label)*100
        test_score=accuracy_score(test_label,self.y_test)*100
        center=km.cluster_centers_
        print("Cluster Centers Coordinate:")
        for i in range(len(center)):
            print("Center",i+1,"=",center[i])
        print(f'Train Score = {train_score:.2f}%')
        print(f'Test Score = {test_score:.2f}%')

    def visualize(self):  #可视化数据
        X,y_kmeans,km=self.X,self.y_kmeans,self.km
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Sepal Width (cm)')
        plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Iris-setosa')
        plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='purple',label='Iris-versicolour')
        plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='yellow',label='Iris-virginica')
        plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=100,c='blue',label='Centroids')
        plt.title('Kmeans Clusters Visualization')
        plt.legend()
        plt.show()

    def elbowMethod(self):
        sse,accuracy=[],[]
        for i in range(1,11):
            kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
            kmeans.fit(self.X_train,self.y_train)
            label=kmeans.predict(self.X_test)
            score=accuracy_score(label,self.y_test)*100
            accuracy.append(score)
            sse.append(kmeans.inertia_)    #sse 平方误差总和
            
        print('\nKmeans SSE:')
        for i in range(len(sse)):
            print("SSE for cluster",i+1,"=",sse[i],end="")
            print(f" | Accuracy = {accuracy[i]:.2f}%")
        plt.plot(range(1,11),sse)
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of the Squared Error (SSE)')   #sse
        plt.show()

    def silhouetteCoefficient(self):
        silhouette_coefficients=[]
        for i in range(2,11):
            kmeans=KMeans(n_clusters=i)
            kmeans.fit(self.X)
            score=silhouette_score(self.X,kmeans.labels_)
            silhouette_coefficients.append(score)
        print('\nKmeans Silhouette Coefficients:')
        for i in range(len(silhouette_coefficients)):
            print("Silhouette Coefficient for cluster",i+2,"=",silhouette_coefficients[i])
        plt.plot(range(2,11),silhouette_coefficients)
        plt.xticks(range(2, 11))
        plt.title('Silhouette Coefficients')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Coefficient')   #sse
        plt.show()

if __name__ == '__main__':
    k=K_Means()
    

