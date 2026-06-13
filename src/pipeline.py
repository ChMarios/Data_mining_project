import os
from data.data_loader import DataLoader
from data.preprocess import Preprocess
from model.modeling import SupervisedLearning
from model.clustering import UnsupervisedLearning
import numpy as np


BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

SAMPLE_SIZE          = 50_000 #None for all
CLUSTER_SAMPLE_SIZE  = 5_000   
CORR_THRESHOLD       = 0.95    
RANDOM_STATE         = 42

def run_pipeline():

    print("--- Starting Pipeline ---")

    print("1. Data Loading")
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()

    print("2. Preprocess")
    proc = Preprocess(df)

    proc.dataset_overview()          
    proc.removing_dupl()
    proc.checking_missing_values()
    proc.statistics()
    proc.data_transformation() 

    #visuals 
    proc.plot_distribution()
    proc.plot_outliers()
    proc.plot_correlation() 

    proc.feature_selection(threshold=CORR_THRESHOLD)
    proc.save_clean_data(PROCESSED_DATA_PATH)

    clean_df = proc.df
 
    if SAMPLE_SIZE and len(clean_df) > SAMPLE_SIZE:
        clean_df = clean_df.sample(n=SAMPLE_SIZE,
                                   random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"\nSampled {SAMPLE_SIZE:,} rows for modelling.")

    X = clean_df.drop('Label', axis=1)
    y = clean_df['Label']

    print("3. SupervisedLearning")

    learner = SupervisedLearning(X, y)
    learner.prepare_data()

    for opt in [1, 2, 3]:          
        learner.run_classification(option=opt)

    print("4.Unsupervised Learning")
    X_scaled  = learner.X_train
    y_labels  = learner.y_train
 
    if len(X_scaled) > CLUSTER_SAMPLE_SIZE:
        idx      = np.random.RandomState(RANDOM_STATE).choice(
                       X_scaled.shape[0], CLUSTER_SAMPLE_SIZE, replace=False)
        X_sample = X_scaled[idx]
        y_sample = y_labels.values[idx]
    else:
        X_sample = X_scaled
        y_sample = y_labels.values
 
    print(f"Clustering on {X_sample.shape[0]:,} samples, "
          f"{X_sample.shape[1]} features.")
 
    unsup = UnsupervisedLearning(X_sample, y_sample)
 
    unsup.optimal_k()
    best_k = unsup.silhouette_analysis()
    print(f"\nRunning K-Means with best k={best_k} (silhouette) and k=3")
    unsup.run_kmeans(k=best_k)
    if best_k != 3:
        unsup.run_kmeans(k=3)       
 
    print("\nRunning Hierarchical Clustering")
    for linkage in ['ward', 'complete', 'average']:
        unsup.run_hierarchical(n_clusters=best_k, linkage=linkage)
 
    print("\nRunning DBSCAN Grid Search")
    best_dbscan_labels, best_cfg = unsup.dbscan_grid_search(
        eps_values=(0.3, 0.5, 1.0, 2.0),
        min_samples_values=(5, 10, 20)
    )



if __name__ == "__main__":
    run_pipeline()