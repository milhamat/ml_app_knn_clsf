import os
import pickle
import numpy as np

class MarketKNN:

    def __init__(self, model_path='artifacts'):
        self.model_path = model_path

    def predict(self, lag1, lag2, lag3, lag4, lag5, vol, tdy):
        try:
            model_path = os.path.join(f"{self.model_path}", "model.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError as e:
            print("Error:", e)
            
        new_val = np.array([[lag1, lag2, lag3, lag4, lag5, vol, tdy]])
        prediction = model.predict(new_val)
        return ''.join(prediction)