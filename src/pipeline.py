import os
from data.data_loader import DataLoader
from data.preprocess import Preprocess
from model.modeling import SupervisedLearning
from model.clustering import UnsupervisedLearning
import numpy as np

def run_pipeline():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, "data", "raw")
    processed_data_path = os.path.join(base_dir, "data", "processed", "cleaned_data.csv")

    print("--- Starting Pipeline ---")

    loader = DataLoader(raw_data_path)
    df = loader.load_data()

    if df is not None:

        proc = Preprocess(df)
        
        proc.removing_dupl()
        proc.checking_missing_values()
        proc.statistics()
        proc.data_transformation()

        # proc.plot_outliers()
        # proc.plot_distribution()
        # proc.plot_correlation()
        proc.feature_selection(threshold=0.95)
        
        proc.save_clean_data(processed_data_path)

        clean_df = proc.df
        # sampling the df 
        clean_df = clean_df.sample(n=100000, random_state=42).reset_index(drop=True)

        X = clean_df.drop('Label', axis=1)
        y = clean_df['Label']

        learner = SupervisedLearning(X,y)

        learner.prepare_data()

        for opt in [1, 2, 3]:
            print(f"\nTraining Model Option: {opt}")
            learner.run_classification(option=opt)

        print("\n--- Starting Unsupervised Phase ---")
        

        X_scaled = learner.X_train 
        y_labels = learner.y_train

        # Sub-sampling 10k for clustering
        if len(X_scaled) > 10000:
            idx = np.random.choice(X_scaled.shape[0], 10000, replace=False)
            X_sample = X_scaled[idx]
            y_sample = y_labels.values[idx] 
        else:
            X_sample = X_scaled
            y_sample = y_labels.values

        unsup = UnsupervisedLearning(X_sample, y_sample)
        
        # K-Means Analysis
        unsup.optimal_k()           
        unsup.silhouette_analysis() 
        unsup.run_kmeans(k=3)       

        # Hierarchical Clustering (Linkage Criteria)
        print("\nRunning Hierarchical Clustering...")
        unsup.run_hierarchical(n_clusters=3, linkage='ward')
        unsup.run_hierarchical(n_clusters=3, linkage='complete')

        # DBSCAN (Density-based)
        print("\nRunning DBSCAN...")
        unsup.run_dbscan(eps=0.5, min_samples=10)
        print("--- Pipeline Completed Successfully ---")
    else:
        print("Pipeline failed: Data not loaded.")

if __name__ == "__main__":
    run_pipeline()