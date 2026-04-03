import os

from data.data_loader import DataLoader
from data.preprocess import Preprocess

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

        proc.plot_outliers()
        proc.plot_distribution()
        proc.plot_correlation()
        proc.feature_selection(threshold=0.95)
        
        proc.save_clean_data(processed_data_path)
        
        print("--- Pipeline Completed Successfully ---")
    else:
        print("Pipeline failed: Data not loaded.")

if __name__ == "__main__":
    run_pipeline()