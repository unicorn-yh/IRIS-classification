import matplotlib.pyplot as plt
import numpy as np
import random

class params:
    def __init__(self,X,y,C,toler,kernel): #toler为容错率
        self.X,self.y,self.C,self.tol=X,y,C,toler
        self.size,self.b=np.shape(X)[0],0
        self.alpha=np.mat(np.zeros((self.size,1)))
        self.error=np.mat(np.zeros((self.size,2)))
        self.K=np.mat(np.zeros((self.size,self.size)))
        for j in range(self.size):
            self.K[:,j]=SMO.Kernel(SMO(),self.X,self.X[j,:],kernel)

class SMO:
    def Kernel(self,X1,X2,kernel):
        SIGMA=1.054
        n=np.shape(X1)[0]
        m=np.mat(np.zeros((n,1)))
        if kernel=='rbf':
            for i in range(n):
                row=X1[i,:]-X2
                m[i]=row*row.T   #||x1-x2||^2 = <x1-x2,x1-x2> = (x1-x2)*(x1-x2).T
            m=np.exp(m/(-1*SIGMA**2))
        else:
            m=X1*X2.T
        return m

    def getRandomIndex(self,index,n):
        i=index
        while(i==index):
            i=int(random.uniform(0,n))
        return i

    def getError(self,param,index):    #超平面方程 f(x)=ΣyiaiK(xi,x)+b
        F_index=float(np.multiply(param.alpha,param.y).T*param.K[:,index]+param.b)
        E_index=F_index-float(param.y[index])   #下标为k的误差 Ek=fk-yk
        return E_index   

    def getNewError(self,param,index):
        error_index=self.getError(param,index)
        param.error[index]=[1,error_index]

    def clipAlpha(self,a,up_bound,down_bound):   
        if a>up_bound: a=up_bound
        elif a<down_bound: a=down_bound
        return a

    def innerLoopHeuristic(self,j,param,Ej):     #内循环启发式 
        '''找符合优化条件的alpha_j(选择条件:使得误差绝对值最大化的alpha_j)'''
        max_error_diff,max_index,new_error=0,-1,0
        param.error[j]=[1,Ej]   #更新误差缓存
        temp=np.nonzero(param.error[:,0].A)[0]    #返回误差不为0的下标
        if len(temp)!=0:
            for i in temp:
                if i==j:
                    continue
                Ei=self.getError(param,i)
                error_diff=abs(Ei-Ej)             #误差绝对值=|Ei-Ej|   
                if(error_diff>max_error_diff):    #更新最大误差
                    max_index=i
                    max_error_diff=error_diff
                    new_error=Ei
            return max_index,new_error
        else:
            index=self.getRandomIndex(j,param.size)
            new_error=self.getError(param,index) 
        return index,new_error

    def optimize_alpha_i_j(self,i,param):
        Ei=self.getError(param,i)  #计算误差
        toler=param.y[i]*Ei        #容错率 toler=yiEi
        alpha=param.alpha[i]
        '''判断alpha是否违反KKT条件,违反则进入循环进行优'''
        if((toler<-param.tol)and(alpha<param.C))or((toler>param.tol)and(alpha>0)):  #违反KKT条件
            '''由innerLoopHeuristic()方法的内层循环选定alpha_j'''
            j,Ej=self.innerLoopHeuristic(i,param,Ei)   #选择条件:使得误差绝对值最大化的alpha_j
            oldAlpha1=param.alpha[i].copy()
            oldAlpha2=param.alpha[j].copy()   
            '''计算上下界'''
            if(param.y[i]!=param.y[j]):  #计算上下界
                dif=param.alpha[j]-param.alpha[i]
                up_bound=max(0,dif)
                down_bound=min(param.C,param.C+dif)
            else:
                sum=param.alpha[j]+param.alpha[i]
                up_bound=max(0,sum-param.C)
                down_bound=min(param.C,sum)
            if up_bound==down_bound:        #已经没有优化的空间
                print("upper bound == lower bound")
                return 0
            '''计算 eta'''
            eta=param.K[i,i]+param.K[j,j]-2*param.K[i,j]
            if eta<=0:     # η=K(xi,xi)+K(xj,xj)-2K(xi,xj)
                return 0
            '''更新 alpha_j'''
            dif1=param.alpha[i]-oldAlpha1   
            dif2=param.alpha[j]-oldAlpha2
            step=param.y[j]*(Ei-Ej)/eta     #步长公式 step=yj(Ei-Ej)/η
            param.alpha[j]+=step            #更新alpha公式 alpha_new=alpha_old+step
            param.alpha[j]=self.clipAlpha(param.alpha[j],up_bound,down_bound)  #修剪新alpha_j
            self.getNewError(param,i)
            if(abs(dif2)<10e-5):            #判断步长是否足够大
                print("Step is too small.")
                return 0
            '''更新 alpha_i'''
            step=param.y[i]*param.y[j]*(-dif2)   #步长公式 step=yiyj(旧alpha_j-新alpha_j)
            param.alpha[i]+=step                 
            self.getNewError(param,i)
            '''更新 b b1 b2'''
            #b1=b_old-Ei-yi(新ai-旧ai)K(xi,xi)-yj(新aj-旧aj)K(xi,xj)
            #b2=b_old-Ej-yi(新ai-旧ai)K(xi,xj)-yj(新aj-旧aj)K(xj,xj)
            b1=param.b-Ei-(param.y[i]*dif1*param.K[i,i])-(param.y[j]*dif2*param.K[i,j])
            b2=param.b-Ej-(param.y[i]*dif1*param.K[i,j])-(param.y[j]*dif2*param.K[j,j])
            if 0<param.alpha[i]<param.C:
                param.b=b1
            elif 0<param.alpha[j]<param.C:
                param.b=b2
            else:
                param.b=(b1+b2)/2
            return 1
        else:
            return 0

    def optimize_all_alpha(self,X,y,C,toler,kernel,maxIter=10000):
        param=params(np.mat(X),np.mat(y).transpose(),C,toler,kernel)
        iter_count,alphaChange,wholeSetData=0,0,True
        while iter_count<maxIter and alphaChange>0 or wholeSetData:
            alphaChange=0
            if wholeSetData:
                for i in range(param.size):   #遍历alpha_i 训练模型
                    alphaChange+=self.optimize_alpha_i_j(i,param)
                    print("Whole Dataset: Iteration",iter_count,"| sample",i,"| alpha change",alphaChange)
                iter_count+=1
            else:
                alpha_in_range=(param.alpha.A>0)*(param.alpha.A<C)
                inside_range=np.nonzero(alpha_in_range)[0]  #0<alpha<C 的下标
                for i in inside_range:
                    alphaChange+=self.optimize_alpha_i_j(i,param)
                    print("Non boundary Dataset: Iteration",iter_count,"| sample",i,"| alpha change",alphaChange)
                iter+=1
            if wholeSetData:
                wholeSetData=False
            elif alphaChange==0:
                wholeSetData=True
            print("Iteration count =",iter_count)
        return param.b,param.alpha

    def trainModel(self):
        kernel='rbf'
        X_train,y_train=getData(train=True)
        b,alpha=self.optimize_all_alpha(X_train,y_train,C=1,toler=0.0001,maxIter=100,kernel=kernel)
        X_train,y_train1=np.mat(X_train),np.mat(y_train).transpose()
        support_vector_index=np.nonzero(alpha.A>0)[0]
        support_vector_X=X_train[support_vector_index]
        support_vector_y=y_train1[support_vector_index]
        print("Number of support vector:",support_vector_X.shape[0])
        size=X_train.shape[0]
        accurate_count=0
        for i in range(size):
            k=self.Kernel(support_vector_X,X_train[i,:],kernel)
            pred=k.T*np.multiply(support_vector_y,alpha[support_vector_index])+b
            if np.sign(pred)==np.sign(y_train[i]):
                accurate_count+=1
        print(f"Train Dataset Accuracy = {accurate_count*100/size:.2f}%")

        X_test,y_test=getData(train=False)
        X_test,y_test=np.mat(X_test),np.mat(y_test).transpose()
        size=X_test.shape[0]
        accurate_count=0
        for i in range(size):
            k=self.Kernel(support_vector_X,X_test[i,:],kernel)
            pred=k.T*np.multiply(support_vector_y,alpha[support_vector_index])+b
            if np.sign(pred)==np.sign(y_test[i]):
                accurate_count+=1
        print(f"Test Dataset Accuracy = {accurate_count*100/size:.2f}%")


def getData(train=True,drawFigure=False):
    data_ls,target_ls,target_name_ls=[],[],[]
    with open('Iris_Project/iris.data','r')as f: #获取文本中的数据
        for line in f:
            temp=line.split(',')
            name=temp[4].strip()
            if not name in target_name_ls:
                target_name_ls.append(name)
            if name==target_name_ls[0]:
                target_ls.append(0)
            elif name==target_name_ls[1]:
                target_ls.append(1)
            elif name==target_name_ls[2]:
                target_ls.append(2)
            numeric_ls=temp[0:4]
            numeric_ls=[float(numeric_string) for numeric_string in numeric_ls]
            data_ls.append(numeric_ls)
    data=np.array(data_ls)
    target=np.array(target_ls)
    rand=np.arange(150)
    np.random.shuffle(rand)
    train_no=rand[:120]
    test_no=np.setdiff1d(rand,train_no)
    X_train,X_test=data[train_no,:2],data[test_no,:2]
    y_train,y_test=target[train_no],target[test_no]
    if drawFigure:
        return data_ls,target_ls,target_name_ls
    if train:
        return list(X_train),list(y_train)
    else:
        return list(X_test),list(y_test)

def showDataSet():   #数据可视化
    X,y,name=getData(train=False,drawFigure=True)
    name1,name2,name0=[],[],[] 
    for i in range(len(X)):
        if y[i]==0:
            name0.append(X[i])
        elif y[i] == 1:
            name1.append(X[i])
        else:
            name2.append(X[i])
    name0=np.array(name0)              
    name1=np.array(name1)
    name2=np.array(name2)           
    plt.scatter(np.transpose(name0)[0], np.transpose(name0)[1],label=name[0])   
    plt.scatter(np.transpose(name1)[0], np.transpose(name1)[1],label=name[1]) 
    plt.scatter(np.transpose(name2)[0], np.transpose(name2)[1],label=name[2]) 
    plt.legend()  
    plt.show()

if __name__=='__main__':
    smo=SMO()
    smo.trainModel()
    showDataSet()












