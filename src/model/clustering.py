from sklearn.cluster import KMeans ,DBSCAN,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")
 
 
def _save_fig(filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=150)
    plt.close()
 
class UnsupervisedLearning():
    
    def __init__(self, X, y):
        self.X = X 
        self.y = y 
        self.pca_data =None

    def optimal_k(self):
        # elbow method fro finding the best k for k-means
        print("Calculating Elbow Method")
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), wcss, marker='x')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Inertia)')
        _save_fig("elbow_method.png")
        plt.show()

    def silhouette_analysis(self):

        print("Silhouette Analysis")
        scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            scores.append(silhouette_score(self.X, cluster_labels, sample_size=5000,
                                           random_state=42))

        best_k = list(range(2,11))[int(np.argmax(scores))]
        print(f"  Best k by silhouette: {best_k}  (score={max(scores):.4f})")


        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 11), scores, marker='o', linestyle='--', color='r')
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        _save_fig("silhouette_analysis.png")
        plt.show()

        return best_k

    def evaluation(self, labels, model_name):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pts  = (labels == -1).sum() if -1 in labels else 0
 
        print(f"  {model_name}")
        print(f"  Clusters: {n_clusters}   Noise points: {noise_pts}")
 
        if n_clusters >= 2:
            # Use sample_size to keep memory manageable
            sil = silhouette_score(self.X, labels, sample_size=10000,
                                   random_state=42)
            db  = davies_bouldin_score(self.X, labels)
            print(f"  Silhouette Score   : {sil:.4f} ")
            print(f"  Davies-Bouldin Idx : {db:.4f}  ")
 
        cm = contingency_matrix(self.y, labels)
        print(cm)
 
        self._plot_pca(labels, model_name)


    def _plot_pca(self, labels, title):
        if self.pca_data is None:
            pca = PCA(n_components=2, random_state=42)
            self.pca_data = pca.fit_transform(self.X)
            print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
 
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1],
                              c=labels, cmap='viridis', alpha=0.4, s=5)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"PCA Visualization – {title}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        safe = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        _save_fig(f"pca_{safe}.png")


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
        self.evaluation(labels, f"DBSCAN (eps={eps}, min_s={min_samples})")
        return labels
    
    def dbscan_grid_search(self, eps_values=(0.3, 0.5, 1.0, 2.0), min_samples_values=(5, 10, 20)):
        results = []
 
        for eps in eps_values:
            for ms in min_samples_values:
                model  = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
                labels = model.fit_predict(self.X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = (labels == -1).mean()
 
                if n_clusters >= 2:
                    sil = silhouette_score(self.X, labels,
                                           sample_size=10000, random_state=42)
                    db  = davies_bouldin_score(self.X, labels)
                else:
                    sil, db = -1.0, 999.0
 
                results.append({
                    'eps': eps, 'min_samples': ms,
                    'n_clusters': n_clusters, 'noise_ratio': noise_ratio,
                    'silhouette': sil, 'davies_bouldin': db
                })
                print(f"  eps={eps}  min_s={ms:2d}"
                      f"clusters={n_clusters}  noise={noise_ratio:.1%}  "
                      f"sil={sil:.4f}  db={db:.4f}")
 
        # Best by Silhouette
        best = max(results, key=lambda r: r['silhouette'])
        print(f"\n  Best DBSCAN config: eps={best['eps']}  "
              f"min_samples={best['min_samples']}  "
              f"silhouette={best['silhouette']:.4f}")
 
        # Run full evaluation on best config
        best_labels = self.run_dbscan(eps=best['eps'],
                                      min_samples=best['min_samples'])
        return best_labels, best

