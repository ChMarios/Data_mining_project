import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Preprocess():

    def __init__(self,df):
        self.df = df.copy()

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

    def statistics(self):
        numeric_stats = self.df.describe().T
        numeric_stats['median'] = self.df.median(numeric_only=True)
        print(numeric_stats)

    def data_transformation(self):

        le = LabelEncoder()
        label = le.fit_transform(self.df["Label"])
        print(self.df["Label"].unique())
        self.df.drop("Label",axis=1,inplace=True)
        self.df["Label"] = label

    def removing_dupl(self) :
        
        print(self.df.duplicated().sum())
        self.df.drop_duplicates(inplace=True)
        print(self.df.duplicated().sum())

    def plot_distribution(self):

        label_counts = self.df['Label'].value_counts()

        print("istribution:")
        print(label_counts)

        plt.figure(figsize=(12,6))

        order = self.df['Label'].value_counts().index
        sns.countplot(data=self.df, y='Label', order=order,hue="Label", palette='viridis',legend=False)
        plt.title('Class Distribution')
        plt.show()

    def plot_outliers(self):
        important_cols = ['Flow Duration', 'Total Fwd Packets', 'Packet Length Mean']

        plt.figure(figsize=(15, 5))
        for i, col in enumerate(important_cols):
            plt.subplot(1, 3, i+1)
            sns.boxplot(y=self.df[col])
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    def plot_correlation(self):
        correlation_matrix = self.df.corr()
        heatmap_segment_size = 15

        num_columns = len(correlation_matrix.columns)
        for i in range(0, num_columns, heatmap_segment_size):
            for j in range(0, num_columns, heatmap_segment_size):
                subset_corr_matrix = correlation_matrix.iloc[i:i+heatmap_segment_size, j:j+heatmap_segment_size]

                plt.figure(figsize=(10, 8))
                sns.heatmap(subset_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title("Correlation Heatmap - Columns {} to {}".format(i+1, i+heatmap_segment_size))
                plt.show()

    def feature_selection(self,threshold = 0.95):
        corr_matrix = self.df.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        self.df.drop(columns=to_drop, inplace=True)
        print(f"Removed {len(to_drop)}")
        return to_drop


    def save_clean_data(self, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Saving cleaned data to {output_path}...")
            self.df.to_csv(output_path, index=False)
            print("File saved successfully!")
        except Exception as e:
            print(f"Error saving file: {e}")