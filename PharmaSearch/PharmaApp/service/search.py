import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
def clusterPharmacy(df):
    scaler = MinMaxScaler()
    df=df[['latitude','longitude']]
    df_scaled=scaler.fit_transform(df[['latitude','longitude']])
    client=scaler.transform([[34.66564845,5.56774134]])
    df_scaled=pd.DataFrame(df_scaled,columns=df.columns)
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
    return df1,client;    
            
def getNearestNeigbors(dataframe,client):
    X=dataframe[['longitude','latitude']]
    y=dataframe["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train) 
    knn.score(X_test,y_test)
    pred=knn.predict(client)
    return pred
    
            
