import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class TrainModel:

    
    def __init__(self, path= "artifacts/Smarket.csv"):
        self.path = path
        
    
    def train(self):
        data = pd.read_csv(self.path)

        new_data = data.drop(columns=["Year", "Unnamed: 0"], axis=1)
        scale = StandardScaler()

        X = pd.DataFrame(scale.fit_transform(new_data.drop("Direction", axis=1),),
                              columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today'])
        y = new_data.Direction
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y,
                                                            test_size=0.3,
                                                            random_state=42,
                                                            stratify=y)
        # create the model
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train.values, y_train.values)
        
        # make predictions
        y_pred = knn.predict(X_test)
        
        # print the model score
        print("model score: {:.2f}".format(knn.score(X_test.values, y_test.values)))
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        # save the model
        model_path = os.path.join("artifacts", "models", "model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path,'wb') as f:
            pickle.dump(knn, f)
        
        
        