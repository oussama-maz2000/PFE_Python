import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from django.db import connection 

crsr = connection.cursor()

def scaler():
    dataframe=pd.read_sql_query("select * from pharmacy",connection)
    dataframe.to_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/original.csv",index=False)
    scaledinfo=calculateScaler(dataframe,dataframe[['latitude']],dataframe[['longitude']])
    lat=np.array(scaledinfo[0][0])
    long=np.array(scaledinfo[0][1])
    data=np.concatenate((lat,long),axis=1)
    df = pd.DataFrame(data, columns = ['latitude','longitude'])
    #df.to_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/scaled.csv",index=False)    
    return df

def clusterPharmacy():
    dataframe=pd.read_sql_query("select * from pharmacy",connection)
    dataframe.to_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/original.csv",index=False)
    ids=dataframe[['id']]
    df1=scaler()
    clustering_model_no_cluster = AgglomerativeClustering()
    clustering_model_no_cluster.fit(df1[['latitude','longitude']])
    df1['cluster'] = clustering_model_no_cluster.labels_
    hist,bin_edges= np.histogram(df1['cluster'],bins=2)
    b=True
    nClust=2
    j=0
    print(hist)
    while b:
        n=nClust 
        for i in range(0,n):
            if (hist[i]>30):
                dft2=df1[df1['cluster']==i]
                clt = AgglomerativeClustering()
                clt.fit(dft2[['latitude','longitude']])
                dft2['cluster'] = (clt.labels_*(nClust))+((clt.labels_-1)*dft2['cluster']*-1)
                df1[df1['cluster']==i]=dft2
                nClust +=1
                j +=1
        hist,bin_edges= np.histogram(df1['cluster'],bins=nClust)  
        if (hist.max()<=30):
            b=False
        df1['id']=ids
        df1.to_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/pharmacyClusters.csv",index=False)    
            
def getNearestNeigborsClient(latitude,longitude):
    pharmacyClusters=pd.read_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/pharmacyClusters.csv")
    dataframe=pd.read_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/original.csv")
    infoScaled=calculateScaler(dataframe,latitude,longitude)
    X=pharmacyClusters[['latitude','longitude']]
    y=pharmacyClusters["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train) 
    knn.score(X_test,y_test)
    pred=knn.predict(infoScaled) 
    print(infoScaled)
    return pred 
    
def setClusterInDB():
    pharmacyClusters=pd.read_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/pharmacyClusters.csv")
    df=pd.read_sql_query("select * from pharmacy",connection)
    cluster=pharmacyClusters['cluster']
    ids=pharmacyClusters['id']
    query='update pharmacy set cluster =%s where (id = %s)'
    for (i,j) in zip(ids,cluster):
        dt=(j,i)
        crsr.execute(query,dt)
    connection.commit() 


def calculateScaler(dataframe,latitude,longitude): 
    minlatitude=dataframe['latitude'].min()
    minlongitude=dataframe['longitude'].min()
    maxlatitude=dataframe['latitude'].max()
    maxlongitude=dataframe['longitude'].max()
    latitudeScaled=(latitude-minlatitude)/(maxlatitude-minlatitude)
    longitudeScaled=(longitude-minlongitude)/(maxlongitude-minlongitude)
    infoScaled=[[latitudeScaled,longitudeScaled]]
    return infoScaled

def ClusterAndClassifyToDB():
    clusterPharmacy()
    setClusterInDB()
            



    
    