from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

class K_Means():
    def __init__(self): 
        self.getData()
        self.elbowMethod()
        cluster_num=3       #从 elbowMethod() 方法可知最佳簇数=3
        self.visualize()
        self.InitCentrePoint(self.X,cluster_num)
        self.iteration(self.X,10)
        self.getScore()
        self.comparison()

    def getData(self):
        iris=load_iris()
        self.X=iris.data[:,2:4]
        self.y=iris.target

    def InitCentrePoint(self,x,cluster_num): 
        '''初始化中心点，中心点个数=cluster_num
        操作：找出距离其它中心点距离之和最远的数据点'''
        print("K-means Kernel for IRIS Dataset")
        self.init_centers=[]
        rand=np.random.randint(len(x))   #随机选第一个中心点
        self.init_centers.append(x[rand])
        for i in range(cluster_num-1):
            distances=[]        
            for j in self.init_centers:    #第一循环有一个中心点，第二循环有两个中心点
                tempDistance=[]  #暂时存储当前数据点到每个中心点的距离
                for k in x:      #对于每个数据点
                    dist=np.sqrt(np.sum(np.square(np.array(k)-np.array(j)))) #计算它们之间的距离
                    tempDistance.append(dist)    #所有数据点到每个中心点的长度的列表，长度：150
                distances.append(tempDistance)   #长度[2][150]的列表
            sumDist=[0 for j in range(len(x))]   #长度150
            for j in distances:   #第一次循环, distance长度1, 第二次循环, distance长度2, j长度150
                sumDist=np.array(sumDist)+np.array(j)   #第一次循环, sumDist=j, 第二次循环, sumDist=j1+j2
            maxDist=max(sumDist) #与中心点距离最远的点
            maxCoordinate=x[list(sumDist).index(maxDist)]  #找到该点的坐标
            self.init_centers.append(maxCoordinate)   #将该坐标加入新的中心点
        
    def EM(self,x,centers,iter_num):   
        '''EM step: 
        E: 将每个数据点分配到离它最近的中心点的簇
        M: 更新簇内所有数据点坐标的均值作为新中心点'''
        new_centers,cluster_of_point=[],[]
        clusters=[[]for i in range(len(centers))]  
        for i in x:   #对于每个数据点
            distances=[]
            for j in centers:
                dist=np.sqrt(np.sum(np.square(np.array(j)-np.array(i))))
                distances.append(dist)   #长度3
            minDist=min(distances)   #该点与中心点最短的距离
            minCoordinate=distances.index(minDist)   #该点的下标: 0/1/2
            cluster_of_point.append(minCoordinate)   #簇分类的数组, 长度150, 值: 0/1/2
            clusters[minCoordinate].append(i)     #簇的数组, 长度3 
        for cluster in clusters:  #长度3
            X,Y=0,0
            for c in cluster:   #长度不定
                X+=c[0]    #该簇内所有X坐标之和 
                Y+=c[1]    #该簇内所有Y坐标之和 
            if len(cluster)==0:    #该簇内没有数据点
                new_centers.append(centers[clusters.index(cluster)])  #该簇的中心点保持不变
            else:
                new_centers.append([round(X/len(cluster),4),round(Y/len(cluster),4)])  #取所有数据点坐标的平均值
        if iter_num==0:
            print("Initialization :",new_centers)
        else:
            print("Iteration",iter_num,":",new_centers)
        self.cluster_of_point,self.clusters,self.new_centers=cluster_of_point,clusters,new_centers

    def iteration(self,x,max_iter):
        self.EM(x,self.init_centers,0)
        for iter_num in range(max_iter):
            self.EM(x,self.new_centers,iter_num+1)
            # self.iter_visualize(iter_num)

    def getScore(self):
        ARI=metrics.adjusted_rand_score(self.y,self.cluster_of_point)*100
        print(f'Adjusted Rand Index = {ARI:.2f}%')  #兰德指数

    def visualize(self):  #可视化数据
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        X,Y,centers=[0,0,0],[0,0,0],[0,0,0]
        for i in range(len(self.y)):
            if self.y[i]==0:
                X[0]+=self.X[i][0]
                Y[0]+=self.X[i][1]
                setosa=plt.scatter(self.X[i][0],self.X[i][1],s=100,c='red')
            if self.y[i]==1:
                X[1]+=self.X[i][0]
                Y[1]+=self.X[i][1]
                versicolour=plt.scatter(self.X[i][0],self.X[i][1],s=100,c='purple')
            if self.y[i]==2:
                X[2]+=self.X[i][0]
                Y[2]+=self.X[i][1]
                virginica=plt.scatter(self.X[i][0],self.X[i][1],s=100,c='yellow')
        for i in range(3):
            centers[i]=round((X[i]/50),4),round((Y[i]/50),4)
            center=plt.scatter(centers[i][0],centers[i][1],s=100,c='blue')
        self.original_centers=centers
        plt.title('Original Data Visualization')
        plt.legend([setosa,versicolour,virginica,center],['Iris-setosa','Iris-versicolour','Iris-virginica','Centroid'])
        #plt.show()

    def iter_visualize(self,iter_num):
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        for cluster in self.clusters:
            for c in cluster:
                if self.clusters.index(cluster)==0: 
                    sec2=plt.scatter(c[0],c[1],s=100,c='purple')
                elif self.clusters.index(cluster)==1: 
                    sec1=plt.scatter(c[0],c[1],s=100,c='red')
                else:
                    sec3=plt.scatter(c[0],c[1],s=100,c='yellow')
        for center in self.new_centers:
            centriod=plt.scatter(center[0],center[1],s=100,c='blue')
        plt.title('Kmeans Kernel Iteration '+str(iter_num+1))
        plt.legend([sec1,sec2,sec3,centriod],['Iris-setosa','Iris-versicolour','Iris-virginica','Centroid'])
        plt.show()
    
    def comparison(self):
        print('\nComparison:')
        for i in range(3):
            self.original_centers[i]=list(self.original_centers[i])
        self.original_centers.sort()
        self.new_centers.sort()
        print('Original Data Centroids =',self.original_centers)
        print('K-means Cluster Centroids =',self.new_centers)
        inertia=sum(((self.original_centers[l]-X)**2).sum() for X, l in zip(self.X,self.y))
        print(f"Original Data SSE = {inertia:.4f}",end="")
        inertia=sum(((self.new_centers[l]-X)**2).sum() for X, l in zip(self.X,self.y))
        print(f" | K-means Cluster SSE = {inertia:.4f}")
    
    def elbowMethod(self):
        sse=[]
        for i in range(1,11):
            kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
            kmeans.fit(self.X,self.y)
            #sse.append(sum(np.min(cdist(self.X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / self.X.shape[0])
            sse.append(kmeans.inertia_)    #sse 平方误差总和
        plt.plot(range(1,11),sse)
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of the Squared Error (SSE)')   #sse
        plt.show()

if __name__ == '__main__':
    k=K_Means()
    

