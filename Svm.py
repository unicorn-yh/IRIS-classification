import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score
import os
import sys
import matplotlib.pyplot as plt
from time import sleep

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.environ.get("_MEIPASS2",os.path.abspath("."))
    return os.path.join(base_path, relative_path)

class dataProcess:
    def __init__(self):
        self.iris={}
        data_ls=[]
        target_ls=[]
        target_name_ls=[]
        with open(resource_path('iris.data'),'r')as f: #获取文本中的数据
            for line in f:
                temp=line.split(',')
                numeric_ls=temp[0:4]
                numeric_ls=[float(numeric_string) for numeric_string in numeric_ls]
                data_ls.append(numeric_ls)
                name=temp[4].strip()
                if not name in target_name_ls:
                    target_name_ls.append(name)
                if name==target_name_ls[0]:
                    target_ls.append(0)
                elif name==target_name_ls[1]:
                    target_ls.append(1)
                elif name==target_name_ls[2]:
                    target_ls.append(2)
        self.data=np.array(data_ls)                  #鸢尾花花瓣和花萼数据
        self.target_names=np.array(target_name_ls)   #鸢尾花类型名称
        self.target=np.array(target_ls)              #鸢尾花类型标签
        self.iris['data']=self.data
        self.iris['target_names']=self.target_names
        self.iris['target']=self.target


    def trainModel(self,st,degree=3,C=1.0):   #训练模型
        if st.lower()=='sepal':
            X=self.data[:,:2]            #前两维数组为花萼数据
        elif st.lower()=='petal':        
            X=self.data[:,2:4]           #后两维数组为花瓣数据
        y=self.target
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test

        #C为正则化参数
        #训练四种不同种类的模型：线性核，径向基核、多项式核、线性函数
        clf=SVC(kernel='linear',C=C,random_state=1).fit(X_train,y_train)  
        rbf_clf=SVC(kernel='rbf',C=C,gamma=0.7,random_state=1).fit(X_train,y_train)
        poly_clf=SVC(kernel='poly',C=C,degree=degree,random_state=1).fit(X_train,y_train)
        lin_clf=LinearSVC(C=C,random_state=1).fit(X_train,y_train)
        clf_dict={}
        clf_dict['SVC with linear kernel']=clf
        clf_dict['LinearSVC (linear kernel)']=lin_clf
        clf_dict['SVC with RBF kernel']=rbf_clf
        clf_dict['SVC with polynomial kernel (degree '+str(degree)+')']=poly_clf
        self.clf_dict=clf_dict

        #四种模型的预测值
        pred=clf.predict(X_test)
        rbf_pred=rbf_clf.predict(X_test)
        poly_pred=poly_clf.predict(X_test)
        lin_pred=lin_clf.predict(X_test)
        pred_dict={}
        pred_dict['SVC with linear kernel']=pred
        pred_dict['LinearSVC (linear kernel)']=lin_pred
        pred_dict['SVC with RBF kernel']=rbf_pred
        pred_dict['SVC with polynomial kernel (degree '+str(degree)+')']=poly_pred
        self.pred_dict=pred_dict

        '''评估指标'''
        #Accuracy Score 准确率
        score=accuracy_score(self.y_test,pred)
        rbf_score=accuracy_score(y_test,rbf_pred)
        poly_score=accuracy_score(y_test,poly_pred)
        lin_score=accuracy_score(y_test,lin_pred)       
        score_dict={}
        score_dict['SVC with linear kernel']=score
        score_dict['LinearSVC (linear kernel)']=lin_score
        score_dict['SVC with RBF kernel']=rbf_score
        score_dict['SVC with polynomial kernel (degree '+str(degree)+')']=poly_score
        self.score_dict=score_dict

        #Cross Validation Score 交叉验证
        CVscore=cross_val_score(clf,X_train,y_train,cv=4)
        rbf_CVscore=cross_val_score(rbf_clf,X_train,y_train,cv=4)
        poly_CVscore=cross_val_score(poly_clf,X_train,y_train,cv=4)
        lin_CVscore=cross_val_score(lin_clf,X_train,y_train,cv=4)       
        cv_score={}
        cv_score['SVC with linear kernel']=CVscore
        cv_score['LinearSVC (linear kernel)']=lin_CVscore
        cv_score['SVC with RBF kernel']=rbf_CVscore
        cv_score['SVC with polynomial kernel (degree '+str(degree)+')']=poly_CVscore
        self.cv_score=cv_score

        #Cross Validation Mean 交叉验证平均值
        cvm_score={}
        cvm_score['SVC with linear kernel']=CVscore.mean()
        cvm_score['LinearSVC (linear kernel)']=lin_CVscore.mean()
        cvm_score['SVC with RBF kernel']=rbf_CVscore.mean()
        cvm_score['SVC with polynomial kernel (degree '+str(degree)+')']=poly_CVscore.mean()
        self.cvm_score=cvm_score

        #Cross Validation Standard Deviation 交叉验证标准差
        cvStd_score={}
        cvStd_score['SVC with linear kernel']=CVscore.std()
        cvStd_score['LinearSVC (linear kernel)']=lin_CVscore.std()
        cvStd_score['SVC with RBF kernel']=rbf_CVscore.std()
        cvStd_score['SVC with polynomial kernel (degree '+str(degree)+')']=poly_CVscore.std()
        self.cvStd_score=cvStd_score


    def getScore(self,st):       #输出所有评估指标
        if st.lower()=='sepal':
            print("\n{0:*^60}".format('SEPAL'))
        else:
            print("\n{0:*^60}".format('PETAL'))
        try:
            print("{0:-^60}".format('ACCURACY SCORE'))
            for key in self.score_dict.keys():
                print(key,":",self.score_dict[key])

            print("\n{0:-^60}".format('CROSS VALIDATION SCORE'))
            for key in self.cv_score.keys():
                print(key,":",self.cv_score[key])

            print("\n{0:-^60}".format('CROSS VALIDATION MEAN'))
            for key in self.cvm_score.keys():
                print(key,":",self.cvm_score[key])

            print("\n{0:-^60}".format('CROSS VALIDATION STANDARD DEVIATION'))
            for key in self.cvStd_score.keys():
                print(key,":",self.cvStd_score[key])
            print("\n") 
        except Exception as e:
            print(e.message)

    def getBestModel(self,st):
        max=0
        for key in self.cvm_score.keys():
            score=self.cvm_score[key]
            if score>max:
                max=score
                self.best_model=key
        print("Best Model for Iris",st,"dataset: ",self.best_model,"\n")

    def getIris(self):  
        return self.iris
    def getData(self):            #长度：600
        return self.data   
    def getTarget(self):          #长度：150
        return self.target
    def getTargetNames(self):     #长度：3
        return self.target_names
    def getPredict(self):
        return self.pred_dict
    def getClf(self):
        return self.clf_dict
    
    def getFigure(self,st,degree=3,C=0.8):
        if st.lower()=='sepal':
            data=self.data[:,:2]
        elif st.lower()=='petal':
            data=self.data[:,2:4]
        y=self.target

        svc_clf=SVC(kernel='linear',C=C).fit(data,y)  
        rbf_clf=SVC(kernel='rbf',C=C,gamma=0.7).fit(data,y)
        poly_clf=SVC(kernel='poly',C=C,degree=degree).fit(data,y)
        lin_clf=LinearSVC(C=C).fit(data,y)

        title=[]
        for t in self.clf_dict.keys():
            title.append(t)             #图表标题
        grid_length=0.02                #网格中的步长

        #创建网格，以绘制图表
        minX,maxX=float(min(data[:,0]))-1,float(max(data[:,0]))+1  #3.3,8.9
        minY,maxY=float(min(data[:,1]))-1,float(max(data[:,1]))+1  #1.0,5.4
        #print(minX,maxX,minY,maxY)
        x_num,y_num=np.meshgrid(np.arange(minX,maxX,grid_length),np.arange(minY,maxY,grid_length))

        for i,clf in enumerate((svc_clf,lin_clf,rbf_clf,poly_clf)):
            plt.subplot(2,2,i+1)
            plt.subplots_adjust(wspace=0.3,hspace=0.6)
            Z=clf.predict(np.c_[x_num.ravel(),y_num.ravel()])
            Z=Z.reshape(x_num.shape)  #x_num.shape=(220, 280)
            plt.contourf(x_num,y_num,Z,cmap=plt.cm.get_cmap('RdYlBu'),alpha=0.95)  #等高线函数：使用不同颜色划分区域
            plt.scatter(data[:,0],data[:,1],c=y,cmap=plt.cm.get_cmap('RdYlBu'))      #以离散点的形式绘制训练数据
            plt.xlabel(st.title()+" length (cm)")
            plt.ylabel(st.title()+" width (cm)")
            plt.xlim(minX,maxX)
            plt.ylim(minY,maxY)
            #plt.xticks(())
            #plt.yticks(())
            plt.title(title[i])
        plt.show()
  
    def getBestParam(self):    #找出最佳模型的最佳惩罚系数
        #The best value of the inverse regularization strength
        params=[0.0001,0.001,0.01,0.1,1,10,100,1000]
        best_model=self.clf_dict[self.best_model]
        grid=GridSearchCV(best_model,{'C':params})
        grid.fit(self.X_train,self.y_train)
        print("The best value of the inverse regularization strength: ")
        print("Best Parameter:",grid.best_params_,"\tScore:",grid.best_score_)
        print("\n")

        #绘制图像：使用不同惩罚系数的准确率 (拟合程度)
        train_score,test_score=[],[]
        for param in params:
            best_model.C=param
            best_model.fit(self.X_train,self.y_train)
            train_score.append(best_model.score(self.X_train,self.y_train))
            test_score.append(best_model.score(self.X_test,self.y_test))
        plt.semilogx(params,train_score,params,test_score)    
        plt.legend(("train","test"))
        plt.ylabel('Accuracy scores')
        plt.xlabel('C(Inverse regularization strength)')
        plt.title('Graphical hyperparameter optimization')
        plt.show()

        
        

if __name__=='__main__':
    print("SVM Classification. Input 'sepal' or 'petal': (Input -1 to exit)")
    st=input()       #input 'sepal' or 'petal'
    st=st.lower()
    while True:
        try:
            if st=='-1':
                break
            if st=='sepal' or st=='petal':
                iris=dataProcess()
                iris.trainModel(st)
                iris.getScore(st)
                iris.getBestModel(st)
                iris.getBestParam()
                iris.getFigure(st)
                st=""
                break
            else:
                print("SVM Classification. Input 'sepal' or 'petal': (Input -1 to exit)")
                st=input()
                st=st.lower()

        except Exception as e:
            print(e.with_traceback)
            sleep(5)
            os.system("pause")


    

    
