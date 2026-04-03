import pandas as pd
import os
import glob


class DataLoader():

    def __init__(self,file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):

        current_dir = os.getcwd()

        path = os.path.join(current_dir, "..", "data", "raw")
        all_files = glob.glob(os.path.join(path,"*.csv"))

        print("Loading Data")

        list_df = []

        for f in all_files:
            temp_df = pd.read_csv(f, low_memory=False)
            # cleaning from nan data
            temp_df.columns = temp_df.columns.str.strip()
            
            
            list_df.append(temp_df)

        self.df = pd.concat(list_df, ignore_index=True)
        self.df.head(20)
        
        return self.df


