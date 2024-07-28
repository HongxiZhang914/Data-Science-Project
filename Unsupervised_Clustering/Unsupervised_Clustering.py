import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

def cluster(df,*args, number_cluster):
    columns = []
    for item in args:
        columns.append(item)
    #train our model
    kmodel = KMeans(n_clusters=number_cluster).fit(df[columns])
    # Put this data back in to the main dataframe corresponding to each observation
    df['Cluster'] = kmodel.labels_
    # Let' visualize these clusters
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.show()
    return df

def elbow(df, start_range, end_range, *args):
    # try using a for loop
    k = range(start_range,end_range)
    K = []
    WCSS = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[column])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    # Store the number of clusters and their respective WSS scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()
    return df

def silhouette(df, start_range, end_range, *args):
    # same as above, calculate sihouetter score for each cluster using a for loop
    k = range(start_range,end_range)
    K = []
    ss = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[column])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[column], ypred)
        K.append(i)
        ss.append(sil_score)
    # Store the number of clusters and their respective silhouette scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':ss})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()
    return df
    

def main():
    df = pd.read_csv('mall_customers.csv')
    #df = cluster(df, 'Annual_Income','Spending_Score', number_cluster=5)
    #df = elbow(df,3,9,'Annual_Income','Spending_Score')
    df = silhouette(df,3,9,'Annual_Income','Spending_Score')

if __name__ =="__main__":
    main()