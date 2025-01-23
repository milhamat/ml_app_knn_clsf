import pickle
import numpy as np

class MarketKNN:

    def __init__(self, model_path='artifacts/model.pkl'):
        #self.path = 'D:/STUDY/Self Study/3.MachineLearning_and_DataScience/Python/x.Project/ml_app_knn_clsf/'
        #self.data = pd.read_csv(self.path + "Smarket.csv")
        self.model_path = model_path

    def predict(self,lag1, lag2, lag3, lag4, lag5, vol, tdy):

        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        new_val = np.array([[lag1, lag2, lag3, lag4, lag5, vol, tdy]])
        prediction = model.predict(new_val)
        return ''.join(prediction)