from sklearn.cluster import KMeans ,DBSCAN,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class UnsupervisedLearning():
    
    def __init__(self, X, y):
        self.X = X 
        self.y = y 
        self.pca_data =None

    def optimal_k(self):
        # elbow method fro finding the best k for k-means
        print("Calculating Elbow Method...")
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), wcss, marker='x')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Inertia)')
        plt.show()

    def silhouette_analysis(self):

        scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.X)
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            scores.append(silhouette_avg)

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 11), scores, marker='o', linestyle='--', color='r')
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.savefig('silhouette_analysis.png')
        plt.show()

    def evaluation(self, labels, model_name):

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        print(f"\nEvaluation: {model_name}")
        if n_clusters < 2:
            print("Not enough clusters for metrics.")
        else:
            sil = silhouette_score(self.X, labels, sample_size=10000)
            db = davies_bouldin_score(self.X, labels)
            print(f"Silhouette Score: {sil:.4f}")
            print(f"Davies-Bouldin Index: {db:.4f}")

        print("Contingency Matrix :")
        matrix = contingency_matrix(self.y, labels)
        print(matrix)
        
        self.plot_pca(labels, model_name)

    def plot_pca(self, labels, title):

        if self.pca_data is None:
            pca = PCA(n_components=2)
            self.pca_data = pca.fit_transform(self.X)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.pca_data[:, 0], y=self.pca_data[:, 1], hue=labels, palette='viridis', alpha=0.6)
        plt.title(f"PCA Visualization: {title}")
        plt.show()

    def run_kmeans(self, k):

        model = KMeans(n_clusters=k,random_state=42, n_init=10)
        labels = model.fit_predict(self.X)
        self.evaluation(labels, f"K-Means (k={k})")
        return labels
    
    def run_hierarchical(self, n_clusters,linkage = 'ward'):
        model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
        agg_res = model.fit_predict(self.X)
        self.evaluation(agg_res, f"Hierarchical ({linkage})")
        return agg_res
    
    def run_dbscan(self, eps=0.5, min_samples=10):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X)
        self.evaluation(labels, "DBSCAN")
        return labels
