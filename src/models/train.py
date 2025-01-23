import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class TrainModel:


    # def __init__(self, lag1, lag2, lag3, lag4, lag5, vol, tdy):
    #     self.lag1 = 0
    #     self.lag2 = 0
    #     self.lag3 = 0
    #     self.lag4 = 0
    #     self. lag5 = 0
    #     self.vol = 0
    #     self.tdy = 0
    #     #self.path = 'D:/STUDY/Self Study/3.MachineLearning_and_DataScience/Python/x.Project/ml_app_knn_clsf/'
    #     #self.data = pd.read_csv(self.path + "Smarket.csv")
    #     self.path = "./Smarket.csv"
    
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


        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train.values, y_train.values)

        # new_val = np.array([[self.lag1, self.lag2, self.lag3, self.lag4, self.lag5, self.vol, self.tdy]])

        # prediction = knn.predict(new_val)
        y_pred = knn.predict(X_test)

        # print(''.join(prediction))
        print("model score: {:.2f}".format(knn.score(X_test.values, y_test.values)))
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        with open('artifacts/model.pkl','wb') as f:
            pickle.dump(knn, f)
        
        
        