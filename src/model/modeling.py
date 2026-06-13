from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "figures")

def _save_fig(filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=150)
    plt.close()
 
 
MODEL_NAMES = {1: "Logistic Regression", 2: "Decision Tree", 3: "Random Forest"}


class SupervisedLearning():

    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_val, self.y_val = None, None
        

    def prepare_data(self):
        #splitting data into train,test,val sets
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=42,stratify=self.y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,stratify=y_temp)

        #prepare data Normalization etc
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_val = scaler.transform(X_val)
        self.X_test = scaler.transform(X_test)

        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test


    def evaluate(self, model, title, option):
        predictions = model.predict(self.X_test)
        model_name  = MODEL_NAMES[option]
 
        print(f"\n{'='*60}")
        print(f"  {title}  |  {model_name}")
        print(f"  Best params: {model.best_params_}")
        print(f"{'='*60}")
        print(classification_report(self.y_test, predictions, zero_division=0))
 
        # Confusion matrix heatmap
        cm = confusion_matrix(self.y_test, predictions)
        cm_norm = confusion_matrix(self.y_test, predictions, normalize='true')
        class_names = getattr(self, 'label_classes_', np.unique(self.y_test))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm, 
            annot=cm,         
            fmt='d',           
            cmap='Blues',      
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar_kws={'label': 'Recall Rate (Normalized)'}
        )
        plt.title(f"Confusion Matrix – {model_name}\n({title.replace('_', ' ')})", fontsize=14, fontweight='bold')
        plt.ylabel("True Label", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        fname = f"cm_{option}_{safe_title}.png"
        _save_fig(fname)
        print(f"Confusion matrix saved {fname}")
 
        return model.best_params_


    def get_grid(self, option):
        if option == 1:   # Logistic Regression
            return {'C': [0.01, 0.1, 1, 10],
                    #'solver' : ['saga'],
                    'solver': ['lbfgs'],
                    'max_iter': [1000]}
        elif option == 2: # Decision Tree
            return {'max_depth': [5, 10, 20, None],
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 10]}
        elif option == 3: # Random Forest
            return {'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]}

    
    def select_classifier(self,option):
        
        if option == 1:
            classifier = LogisticRegression(solver='lbfgs',max_iter=1000, class_weight='balanced')
        elif option == 2:
            classifier = DecisionTreeClassifier(class_weight='balanced',random_state=42)
        elif option == 3:
            classifier = RandomForestClassifier(class_weight='balanced', n_jobs=-1,random_state=42)
        else:
            raise ValueError("option values are : 1,2,3")
        
        return classifier

    def run_classification(self, option):
        classifier = self.select_classifier(option)
        params     = self.get_grid(option)
        model_name = MODEL_NAMES[option]
        print(f"  Model: {model_name}")
 
        # Prefined split
        split      = [-1] * len(self.X_train) + [0] * len(self.X_val)
        pds        = PredefinedSplit(test_fold=split)
        X_combined = np.concatenate((self.X_train, self.X_val), axis=0)
        y_combined = np.concatenate((self.y_train, self.y_val), axis=0)
 
        grid_split = GridSearchCV(classifier, params, cv=pds,
                                  scoring="f1_weighted", n_jobs=-1)
        grid_split.fit(X_combined, y_combined)
        best_split = self.evaluate(grid_split, "Train_Val_Split", option)
 
        # 5 fold cross validation
        grid_cv = GridSearchCV(classifier, params, cv=5,
                               scoring='f1_weighted', n_jobs=-1)
        grid_cv.fit(self.X_train, self.y_train)
        best_cv = self.evaluate(grid_cv, "5_fold_CV", option)
 
        # compare best params
        print(f"\n--- Hyper-parameter comparison for {model_name} ---")
        print(f"  Train/Val Split best : {best_split}")
        print(f"  5-fold CV best       : {best_cv}")
        

    