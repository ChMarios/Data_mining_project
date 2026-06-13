import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


FIGURES = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")

def _save_fig(filename):
    os.makedirs(FIGURES, exist_ok=True)
    plt.savefig(os.path.join(FIGURES, filename), bbox_inches="tight", dpi=150)
    plt.close()

class Preprocess():

    def __init__(self,df):
        self.df = df.copy()

    
    def dataset_overview(self):
        print(f"Dataset shape : {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print("\nColumn dtypes:")
        print(self.df.dtypes.value_counts())
        print("\nFirst 5 rows:")
        print(self.df.head())

    def checking_missing_values(self):
        #Identify the number of missing values
        data = self.df

        missing_values = data.isna().sum()
        print(missing_values.loc[missing_values > 0])


        numeric_cols = data.select_dtypes(include=np.number).columns
        count = np.isinf(data[numeric_cols]).sum()
        print(count[count>0])

        #replacing infinity values with Nan
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f'Missing values after processing infinite values: {data.isna().sum().sum()}')
        missing = data.isna().sum()
        print(missing.loc[missing > 0])

        #Remove the missing values
        data.dropna(inplace = True)

        # Fill the missing values with mean
        data["Flow Bytes/s"] = data.get("Flow Bytes/s", pd.Series()).fillna(data.get("Flow Bytes/s", pd.Series()).mean())
        data["Flow Packets/s"] = data.get("Flow Packets/s", pd.Series()).fillna(data.get("Flow Packets/s", pd.Series()).mean())

        # Deleting the rest missing values
        data.dropna(inplace=True)
        Nans_data = data.isnull().sum().sort_values(ascending=False)
        print(Nans_data)
        data.head(20)

    def removing_dupl(self):
        before = self.df.duplicated().sum()
        self.df.drop_duplicates(inplace=True)
        print(f"\nDuplicates removed: {before:,} remaining: {self.df.duplicated().sum()}")


    def statistics(self):
        print("\nStatistics")
        stats = self.df.describe().T
        stats['median'] = self.df.median(numeric_only=True)
        print(stats)
        return stats


    def data_transformation(self):

        le = LabelEncoder()
        print(f"\nLabel classes: {self.df['Label'].unique()}")
        self.df["Label"] = le.fit_transform(self.df["Label"])
        self.label_classes_ = le.classes_   # store for reference in report
        print(f"Encoded as   : {list(range(len(le.classes_)))}")


    def removing_dupl(self) :
        
        print(self.df.duplicated().sum())
        self.df.drop_duplicates(inplace=True)
        print(self.df.duplicated().sum())

    def plot_distribution(self):
        label_counts = self.df['Label'].value_counts()
        print("\nClass Distribution:")
        print(label_counts)
 
        plt.figure(figsize=(12, 6))
        order = label_counts.index
        ax = sns.countplot(data=self.df, y='Label', order=order,
                      hue='Label', palette='viridis', legend=False)
        ax.set_xscale('log')
        plt.title('Class Distribution (Label)')
        plt.xlabel('Count')
        _save_fig("class_distribution.png")


    def plot_outliers(self):

        important_cols = [c for c in
                          ['Flow Duration', 'Total Fwd Packets', 'Packet Length Mean']
                          if c in self.df.columns]
 
        plt.figure(figsize=(5 * len(important_cols), 5))
        for i, col in enumerate(important_cols):
            plt.subplot(1, len(important_cols), i + 1)
            sns.boxplot(y=np.log10(self.df[col].abs() + 1))
            plt.title(f'Boxplot (Log10) – {col}')
            plt.ylabel('Log10 Value')
        plt.tight_layout()
        _save_fig("boxplots_outliers.png")


    def plot_correlation(self, threshold=0.8):
        numeric_df = self.df.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        
        high_corr_features = set()
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold:
                    high_corr_features.add(corr.columns[i])
                    high_corr_features.add(corr.columns[j])
                    
        filtered_corr = corr.loc[list(high_corr_features), list(high_corr_features)]
        
        if not filtered_corr.empty:
            plt.figure(figsize=(14, 11))
            sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f", 
                        annot_kws={"size": 7}, vmin=-1, vmax=1, cbar=True)
            plt.title(f"Filtered Correlation Heatmap (Abs Corr > {threshold})", fontsize=14)
            plt.tight_layout()
            _save_fig("heatmap.png")
 


    def feature_selection(self, threshold=0.95):
        numeric_df = self.df.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)
        print(f"\nFeature selection (threshold={threshold}): removed {len(to_drop)} columns")
        if to_drop:
            print("  Dropped:", to_drop)
        return to_drop



    def save_clean_data(self, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Saving cleaned data to {output_path}...")
            self.df.to_csv(output_path, index=False)
            print("File saved successfully!")
        except Exception as e:
            print(f"Error saving file: {e}")