import pandas as pd
import pandas as pdo
import plotly.express as px
from sklearn.manifold import TSNE 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import scipy
from tabulate import tabulate
import seaborn as sns
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.io as sio

def Q1_1():
    mat = sio.loadmat('dataset_1.mat')
    samples=np.array(mat['samples'])
    labels=np.array(mat['labels'])
    sample=np.empty(shape=(100,28,28),dtype='float')
    p=0
    #print(mat['labels'][0])
    for i in range(10):
    	counter=0
    	for row in range(len(labels[0])):
            if(labels[0][row]==i and counter<10):
                sample[p]=samples[row]
                counter+=1
                p+=1


    fig, axes = plt.subplots(10,10, figsize=(10,10))
    #print(labels)
    for i,ax in enumerate(axes.flat):
    	ax.imshow(np.squeeze(sample[i]))
    #print(samples.shape)
    #print(labels.shape)
    plt.savefig('fig_1_1.jpg')

def Q1_2():
        #Answer 1_2
    mat1 = sio.loadmat('dataset_2.mat')
    l2=np.array(mat1['labels'][0])
    #print(l2)
    x_value=[]
    y_value=[]
    data2=[]

    s2=np.array(mat1['samples'])
    for i in range(len(s2)):
        data2.append([s2[i][0],s2[i][1],int(l2[i])])
    # print(s2)
    rows=np.array(data2)
    columnNames=['x_value','y_value','label']
    dataframe = pd.DataFrame(data=rows, columns=columnNames)
    dataframe['label'] = dataframe['label'].astype(int)

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=dataframe,x='x_value', y='y_value', hue='label',palette="deep")
    plt.legend(loc=4)
    plt.title("Scattered Plot of data",fontsize=20,color="w")
    plt.tight_layout()
    plt.savefig('fig_1_2.jpg')
    plt.show()




def Q1_3():

	#Answer 1_3
    mat = sio.loadmat('dataset_1.mat')
    sample=np.array(mat['samples'])
    result=np.empty(shape=(50000,784),dtype='float')
    result = sample.reshape([50000,784])
    # print(sample.shape)
    # print(result.shape)
    label=np.array(mat['labels'][0])

    model = TSNE(n_components = 2, random_state = 0,perplexity=30,n_iter=1000) 
    tsne_data = model.fit_transform(result)
    tsne_data = np.vstack((tsne_data.T, label)).T 
    tsne_df = pd.DataFrame(data = tsne_data, 
         columns =("Dim_1", "Dim_2", "label"))

    sns.FacetGrid(tsne_df, hue ="label", size = 8).map( 
           plt.scatter, 'Dim_1', 'Dim_2')
    plt.legend(loc=4)
    plt.title("Scattered Plot by reducing dimension using t-SNE",fontsize=20,color="w")
    plt.tight_layout()
    plt.savefig('fig_1_3.jpg')
    plt.show()



def Q1_4():
	#Answer 1_4
    mat4 = sio.loadmat('dataset_1.mat')
    # print(mat['samples'][0])
    #print(mat.shape)
    sample4=np.array(mat4['samples'])
    result4=np.empty(shape=(50000,784),dtype='float')
    result4 = sample4.reshape([50000,784])
    label4=np.array(mat4['labels'][0])
    model = TSNE(n_components = 3, random_state = 0,perplexity=30,n_iter=1000) 
    tsne_data = model.fit_transform(result4)
    tsne_data = np.vstack((tsne_data.T, label4)).T 
    tsne_df = pd.DataFrame(data = tsne_data, 
         columns =("Dim_1", "Dim_2","Dim_3" ,"label"))
    tsne_df['label'] = tsne_df['label'].astype(int)
    tsne_df['label'] = tsne_df['label'].astype(str)
    fig = px.scatter_3d(tsne_df,x="Dim_1",y="Dim_2",z="Dim_3",
                  color='label',title="Scattered Plot by reducing to 3dimension using t-SNE",width=1000, height=800,)
    fig.show()


def Q2_1():
    ############################      ANSWER 2_1     #########################################################
    mat21 = sio.loadmat('dataset_2.mat')
    sample21=np.array(mat21['samples'])
    label21=np.array(mat21['labels'][0])
    msk = np.random.rand(len(label21)) <= 0.7        # Picking upto approx 70% Uniformaly Distributed Data
    train_label = label21[msk]
    test_label = label21[~msk]
    train_sample = sample21[msk]
    test_sample = sample21[~msk]
    # Grid Search for hyperparameter as max_depth 
    arr=[]
    for i in range(2,31):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf=clf.fit(train_sample,train_label)
        y_predict=clf.predict(test_sample)
        c=0
        for x ,y in zip(y_predict,test_label):
            if x==y:
                c+=1
        result=(c*100)/len(test_label)
        arr.append(result)

        for i in range(len(arr)):
            print("Depth {0} Testing Accuracy {1}".format((i+2),arr[i]))
    depth=[x for x in range(2,31)]
    plt.plot(depth,arr,'bo',marker='o',color='g',linestyle='-')
    plt.suptitle('Depth Vs Accuracy in Grid Search')
    plt.xlabel('Depth')
    plt.ylabel('Testing Accuracy(%)')
    plt.savefig('fig2_1.jpg')
    plt.show()



def Q2_2():
    ############################      ANSWER 2_2     #########################################################
    mat21 = sio.loadmat('dataset_2.mat')
    sample21=np.array(mat21['samples'])
    label21=np.array(mat21['labels'][0])
    msk = np.random.rand(len(label21)) <= 0.7        # Picking upto approx 70% Uniformaly Distributed Data
    train_label = label21[msk]
    test_label = label21[~msk]
    train_sample = sample21[msk]
    test_sample = sample21[~msk]
    # Grid Search for hyperparameter as max_depth 
    train_acc=[]
    valid_acc=[]
    for i in range(2,31):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf=clf.fit(train_sample,train_label)
        y_predict_test=clf.predict(test_sample)
        y_predict_train=clf.predict(train_sample)
        c=0
        for x ,y in zip(y_predict_test,test_label):
            if x==y:
                c+=1
        result=(c*100)/len(test_label)
        valid_acc.append(result)
        d=0
        for x ,y in zip(y_predict_train,train_label):
            if x==y:
                d+=1
        result=(d*100)/len(train_label)
        train_acc.append(result)
    depth=[x for x in range(2,31)]
# for i in range(len(arr)):
#   print("Depth {0}  :   Validation Accuracy {1:.4f}   : Training Acuuracy  {2:.4f}  ".format((i+2),valid_acc[i],train_acc[i]))
    df=pd.DataFrame({'Max Depth': depth,
                  'Validation Accuracy':valid_acc,
                 'Training Accuracy' : train_acc   
    })
    print(tabulate(df, headers='keys', tablefmt='psql',showindex='never'))
    df = df.melt('Max Depth', var_name='Accuracy',  value_name='Accuracy(%)')
    g = sns.catplot(x="Max Depth", y="Accuracy(%)", hue='Accuracy', data=df,kind='point',aspect=1.5)
    g.fig.suptitle('Depth Vs Accuracy') 
    g.savefig('fig2_2.jpg')
    plt.show()


def Q2_3():
    ############################      ANSWER 2_3     #########################################################
    mat21 = sio.loadmat('dataset_2.mat')
    sample21=np.array(mat21['samples'])
    label21=np.array(mat21['labels'][0])
    msk = np.random.rand(len(label21)) <= 0.7        # Picking upto approx 70% Uniformaly Distributed Data
    train_label = label21[msk]
    test_label = label21[~msk]
    train_sample = sample21[msk]
    test_sample = sample21[~msk]
    # Grid Search for hyperparameter as max_depth 
    train_acc=[]
    valid_acc=[]
    train_acc_sk=[]
    valid_acc_sk=[]
    for i in range(2,31):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf=clf.fit(train_sample,train_label)
        y_predict_test=clf.predict(test_sample)
        y_predict_train=clf.predict(train_sample)
        c=0
        for x ,y in zip(y_predict_test,test_label):
            if x==y:
                c+=1
        result=(c*100)/len(test_label)
        valid_acc.append(result)
        d=0
        for x ,y in zip(y_predict_train,train_label):
            if x==y:
                d+=1
        result=(d*100)/len(train_label)
        train_acc.append(result)
  # Sklearn Metric for Accuracy Score 
        train_acc_sk.append(100*metrics.accuracy_score(y_predict_train, train_label))
        valid_acc_sk.append(100*metrics.accuracy_score(y_predict_test, test_label))

  

    depth=[x for x in range(2,31)]
# for i in range(len(arr)):
#   print("Depth {0}  :   Validation Accuracy {1:.4f}   : Training Acuuracy  {2:.4f}  ".format((i+2),valid_acc[i],train_acc[i]))
    df1=pd.DataFrame({'Max Depth': depth,
                  'Validation Accuracy User Define':valid_acc,
                 'Valdation Accuracy sklearn' : valid_acc_sk   
    })
    df2=pd.DataFrame({'Max Depth': depth,
                  'Training Accuracy User Defined':train_acc,
                 'Training Accuracy sklearn' : train_acc_sk   
    })
    print(tabulate(df1, headers='keys', tablefmt='psql',showindex='never'))

    print(tabulate(df2, headers='keys', tablefmt='psql',showindex='never'))


def Q3_1():
    ############################   Answer3_1    ##########################################################
    data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv") 
    data=data.drop(['No'],axis=1)
    m=data['pm2.5'].median()
    s=data.median()
    data=data.fillna(data.median())
# print(data.mean())
    col_name=['year','day','hour','pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']

# print(X.size)
# print(Y.size)

    msk = np.random.rand(len(data)) < 0.8

    train = data[msk]
    test = data[~msk]
    X_train=train[col_name]
    Y_train=train['month']
    X_test=test[col_name]
    Y_test=test['month']
#Gini
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)
    gini=metrics.accuracy_score(Y_test, y_pred)
    print("Accuracy: using Gini Index",gini)
#Entropy
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)
    entropy=metrics.accuracy_score(Y_test, y_pred)
    print("Accuracy: using Entropy",entropy)
    if gini>entropy:
        print("Best is Gini Index")
    else:
        print("Best is Entropy")


def Q3_2():
    ############################   Answer3_2   ##########################################################
    data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv") 
    data=data.drop(['No'],axis=1)
    m=data['pm2.5'].median()
    s=data.median()                           
    data=data.fillna(data.median())          ########Fillinfg the NAN with the median value
# print(data.mean())
    col_name=['year','day','hour','pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']
#Splitting of data 
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]

    X_train=train[col_name]
    Y_train=train['month']
    X_test=test[col_name]
    Y_test=test['month']
    depth=[2, 4, 8, 10, 15, 30]
    train_accuracy = list()
    test_accuracy = list()
#Gini
    for i in depth:
        clf = DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X_train,Y_train)
        y_pred = clf.predict(X_test)
        test_accuracy.append(100*(metrics.accuracy_score(Y_test, y_pred)))
        train_accuracy.append(100*(metrics.accuracy_score(Y_train, clf.predict(X_train))))

    for i in range(len(depth)):
        print("For depth {0} the training accuracy ={1} and testing Accuracy= {2}".format(depth[i],train_accuracy[i],test_accuracy[i]))

    df=pd.DataFrame({'Depth':depth,
                 'Training Accuracy':train_accuracy,
                 'Testing Accuracy' : test_accuracy})
    df = df.melt('Depth', var_name='Accuracy',  value_name='Accuracy(%)')
    g = sns.catplot(x="Depth", y="Accuracy(%)", hue='Accuracy', data=df,kind='point')
    g.fig.suptitle('Depth Vs Accuracy') 
    g.savefig('fig3_2.jpg')
    plt.show()

            


def Q3_3():
    ######################### Answer 3_3 ###############################################
    data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv") 
    data=data.drop(['No'],axis=1)
    m=data['pm2.5'].median()
    s=data.median()                           
    data=data.fillna(data.median())          ########Fillinfg the NAN with the median value
# print(data.mean())
    col_name=['year','day','hour','pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']
#Splitting of data 
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]

    X_train=train[col_name]
    Y_train=train['month']
    X_test=test[col_name]
    Y_test=test['month']
# majority2=scipy.stats.mode(Y_test)
    s=int((X_test.size/10))
    count=0
    res=np.empty(shape=(100,s),dtype=int)
    for i in range(100):
        msk = np.random.rand(len(train)) < 0.5  
        train_stamp = train[msk]                  # 50% random sampling with Uniform Distribution
        test_stamp = train[~msk]
        X_train_stamp=train_stamp[col_name]
        Y_train_stamp=train_stamp['month']
        clf = DecisionTreeClassifier(max_depth=3)
        clf = clf.fit(X_train_stamp,Y_train_stamp)
        y_pred = clf.predict(X_test)
        res[count]=y_pred
        count+=1

    arr=scipy.stats.mode(res)
    arr=arr[0]
    z=0
    for x,y in zip(arr[0],Y_test):
        if(x==y):
            z+=1
    print((z/len(arr[0]))*100)


def Q3_4():
    ######################### Answer 3_4 ###############################################
    data = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv")
    data=data.drop(['No'],axis=1)
    m=data['pm2.5'].median()
    s=data.median()                           
    data=data.fillna(data.median())          ########Fillinfg the NAN with the median value
    col_name=['year','day','hour','pm2.5','DEWP','TEMP','PRES','Iws','Is','Ir']
#Splitting of data 
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]

    X_train=train[col_name]
    Y_train=train['month']
    X_test=test[col_name]
    Y_test=test['month']
    testing_acc=[]
    training_acc=[]
    s=int((X_test.size/10))
    s1=int((X_train.size/10))
    # print(s)
    # print(s1)
    # print(X_train.shape)

    tr=[]
    d=[]
    te=[]
    tra=[]
    trees=[20,50,100,150,200,250,300,350]
    depth=[4,8,10,15,20,30]
    for y in range(len(depth)):
        for x in trees:
            res_test=np.empty(shape=(x,s),dtype=int)
            res_train=np.empty(shape=(x,s1),dtype=int)
            count=0
            for i in range(x):
      # print(x)
                msk = np.random.rand(len(train)) < 0.5  
                train_stamp = train[msk]                  # 50% random sampling with Uniform Distribution
                test_stamp = train[~msk]
                X_train_stamp=train_stamp[col_name]
                Y_train_stamp=train_stamp['month']
                clf = DecisionTreeClassifier(max_depth=depth[y])
                clf = clf.fit(X_train_stamp,Y_train_stamp)
                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                res_test[count]=y_pred_test
                res_train[count]=y_pred_train
                count+=1
            arr=scipy.stats.mode(res_test)
            arr=arr[0]
            arr1=scipy.stats.mode(res_train)
            arr1=arr1[0]
            z=0
            for x1,y1 in zip(arr[0],Y_test):
                if(x1==y1):
                    z+=1
    
            z1=0
            for x1,y1 in zip(arr1[0],Y_train):
                if(x1==y1):
                    z1+=1
    # print((z1/len(arr[0]))*100)
            d.append(depth[y])
            tr.append(x)
            te.append((z/len(arr[0]))*100)
            tra.append((z1/len(arr1[0]))*100)

            print("Depth {0} and tree {1} Testing Accuracy :   {2}  Training Accuracy : {3}  ".format(depth[y],x,(z/len(arr[0]))*100,(z1/len(arr1[0]))*100))
  
  
    df=pd.DataFrame({'Max Depth': d,
                 "No. of Trees":tr,
                  'Training Accuracy':tra,
                 'Testing Accuracy' : te   
    })
    print(tabulate(df, headers='keys', tablefmt='psql',showindex='never'))

  


















if __name__ == "__main__":
    print("PhD19006")
    Q1_1()
    Q1_2()
    Q1_3()
    Q1_4()
    Q2_1()
    Q2_2()
    Q2_3()
    Q3_1()
    Q3_2()
    Q3_3()
    Q3_4()