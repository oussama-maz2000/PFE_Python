from http import client
from operator import index, indexOf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from django.db import connection 

crsr = connection.cursor()
def clusterPharmacy():
    dataframe=pd.read_sql_query("select * from PharmaApp_pharmalocation",connection)
    dataframe.to_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service/original.csv",index=False)
    scaler = MinMaxScaler() 
    dataframe=dataframe[['latitude','longitude']]
    df_scaled=scaler.fit_transform(dataframe[['latitude','longitude']])
    df_scaled=pd.DataFrame(df_scaled,columns=dataframe.columns)
    df1=df_scaled
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
        df1.to_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service/pharmacyClusters.csv",index=False)    
        
            
def getNearestNeigbors(latitude,longitude):
    pharmacyClusters=pd.read_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service/pharmacyClusters.csv")
    dataframe=pd.read_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service/original.csv")
    minlatitude=dataframe['latitude'].min()
    minlongitude=dataframe['longitude'].min()
    maxlatitude=dataframe['latitude'].max()
    maxlongitude=dataframe['longitude'].max()
    latitudeScaled=(latitude-minlatitude)/(maxlatitude-minlatitude)
    longitudeScaled=(longitude-minlongitude)/(maxlongitude-minlongitude)
    client=[[latitudeScaled,longitudeScaled]]
    X=pharmacyClusters[['longitude','latitude']]
    y=pharmacyClusters["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train) 
    knn.score(X_test,y_test)
    pred=knn.predict(client) 
    return pred 
    
def setClusterInDB():
    pharmacyClusters=pd.read_csv("/Users/macbookair/Dev/project/pharma-v-recharche/PharmaSearch/PharmaApp/service/pharmacyClusters.csv")
    cluster=pharmacyClusters['cluster']
    query='update PharmaApp_pharmalocation set cluster =%s where (id = %s)'
    for i in range(len(cluster)):
        e=i+1
        element=cluster[i]
        dt=(element,e)
        crsr.execute(query,dt)
    connection.commit()
      

def getNearsetPharmacyFromDB(num):
    query='select * from PharmaApp_pharmalocation where cluster=%s'
    crsr.execute(query,num)
    pharmacy=crsr.fetchall()
    return pharmacy

            
