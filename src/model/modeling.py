from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np

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


    def evaluate(self,model,title):
        predictions = model.predict(self.X_test)
        print(f"\nResults for {title}:")
        print(f"Best Parameters: {model.best_params_}")

        print(classification_report(self.y_test, predictions))
        print("Confusion Matrix :")
        print(confusion_matrix(self.y_test, predictions))

    def get_grid(self,option):
        
        if option == 1: # Logistic Regression
            return {'C': [0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [1000]}
        elif option == 2: # Decision Tree
            return {'max_depth': [10, 20, None], 'criterion': ['gini', 'entropy']}
        elif option == 3: # Random Forest
            return {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    
    def select_classifier(self,option):
        
        if option == 1:
            classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        elif option == 2:
            classifier = DecisionTreeClassifier(class_weight='balanced')
        elif option == 3:
            classifier = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
        else:
            raise ValueError("option values are : 1,2,3")
        
        return classifier

    def run_classification(self,option):

        classifier = self.select_classifier(option)
        params = self.get_grid(option)

        split = [-1] * len(self.X_train) + [0] * len(self.X_val)
        pds = PredefinedSplit(test_fold=split)

        X_combined = np.concatenate((self.X_train, self.X_val), axis=0)
        y_combined = np.concatenate((self.y_train, self.y_val), axis=0)

        grid = GridSearchCV(classifier, params, cv=pds, scoring="f1_weighted")
        grid.fit(X_combined,y_combined)
        self.evaluate(grid,"Senario No CV")

        grid_cv = GridSearchCV(classifier, params, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_cv.fit(self.X_train, self.y_train)
        self.evaluate(grid_cv, "Scenario 2 (With CV)")
    