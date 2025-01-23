import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MarketKNN:


    def __init__(self, lag1, lag2, lag3, lag4, lag5, vol, tdy, model_path='artifacts/model.pkl'):
        self.lag1 = 0
        self.lag2 = 0
        self.lag3 = 0
        self.lag4 = 0
        self. lag5 = 0
        self.vol = 0
        self.tdy = 0
        #self.path = 'D:/STUDY/Self Study/3.MachineLearning_and_DataScience/Python/x.Project/ml_app_knn_clsf/'
        #self.data = pd.read_csv(self.path + "Smarket.csv")
        self.model_path = model_path

    def predict(self):

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        new_val = np.array([[self.lag1, self.lag2, self.lag3, self.lag4, self.lag5, self.vol, self.tdy]])
        prediction = model.predict(new_val)
        return ''.join(prediction)