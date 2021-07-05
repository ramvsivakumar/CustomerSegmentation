import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

def plot_histogram(data, bins, xlabel, ylabel, title):

    N, bins, patches = plt.hist(data, bins = bins); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    frac = N/N.max()
    norm = colors.Normalize(frac.min(), frac.max())
    viridis = plt.cm.get_cmap('viridis',12)

    for afrac, apatch in zip(frac, patches):
        color = viridis(norm(afrac))
        apatch.set_facecolor(color)
        

def clustering(data, clusters):

    kmeans = KMeans(n_clusters=clusters, random_state=10)
    labels = kmeans.fit_predict(data)

    sil_score_avg = silhouette_score(data, labels)
    print(sil_score_avg)
    sil_sample = silhouette_samples(data, labels)

    return kmeans, labels, sil_score_avg, sil_sample


def visualize_clusters(data, dimension_reduction = False):

    for clusters in range_clusters:

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(121)
        ax1.set_xlim([-0.1,1])
        ax1.set_ylim(0, len(data) + (clusters+1)*10)

        kmeans, labels, sil_score_avg, sil_sample = clustering(data, clusters)

        y_low = 10
        for i in range(clusters):

            sil_values_cluster = sil_sample[labels == i]
            sil_values_cluster.sort()

            size = sil_values_cluster.shape[0]
            y_high = size + y_low

            color = plt.cm.nipy_spectral(float(i) / clusters)
            ax1.fill_betweenx(np.arange(y_low, y_high), 0, sil_values_cluster,
                              facecolor = color, edgecolor = color, alpha=0.7)
            ax1.text(-0.05, y_low + size, str(i))
            y_low = y_high+10

        ax1.set_title("Silhouette plot for various clusters")
        ax1.set_xlabel("Silhouette score")
        ax1.set_ylabel("Clusters")

        ax1.axvline(x=sil_score_avg, color="blue", linestyle="--")
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


        colors = plt.cm.nipy_spectral(labels.astype(float)/clusters)

        if not dimension_reduction:

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], marker='.',
                        s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

            centers = kmeans.cluster_centers_
            for i,c in enumerate(centers):
                ax2.scatter(c[0], c[1], c[2], marker = '$%d$' % i, alpha=1, s=30, edgecolor='k')

            ax2.set_title("Cluster Visualization")
            ax2.set_xlabel("Age")
            ax2.set_ylabel("Annual Income")
            ax2.set_zlabel("Spending")

        else:

            ax2 = fig.add_subplot(122)
            ax2.scatter(data[:, 0], data[:, 1], marker='.',
                        s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

            centers = kmeans.cluster_centers_
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=30, edgecolor='k')

            ax2.set_title("Cluster Visualization after PCA")
            ax2.set_xlabel("First Principal Component")
            ax2.set_ylabel("Second Principal Component")

    plt.show()

def main():

    data = pd.read_csv("customer-segmentation-dataset/customer-segmentation-dataset/Mall_Customers.csv")

    plt.pie([data.Gender.value_counts()[1],data.Gender.value_counts()[0]],labels=list(set(data.Gender.values)),autopct='%1.1f%%', explode=[0.1,0])

    # data['Gender'].replace('Female',0,inplace=True)
    # data['Gender'].replace('Male',1,inplace=True)

   
    plot_histogram(data['Annual Income (k$)'], 14, 'Annual Income', 'Count', 'Histogram of Annual Income')

    plot_histogram(data['Age'], 10, 'Age', 'count', 'Histogram of Age')

    plot_histogram(data['Spending Score (1-100)'], 10, 'Spending Score', 'count', 'Histogram of Spending Score')

    range_clusters = [2, 3, 4, 5, 6]
    data_subset = data.iloc[:, 2:5]    

    visualize_clusters(data_subset, dimension_reduction=False)

    pca = PCA(n_components=2)
    pca_reduced = pca.fit_transform(data_subset)
    print(pca.explained_variance_ratio_)

    visualize_clusters(pca_reduced, dimension_reduction=True)

if __name__ == '__main__':

    main_fn = main()
    t = threading.Thread(target=main_fn)
    t.setDaemon(True)
    t.start()




