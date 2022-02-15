import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MarketKNN:


    def __init__(self, lag1, lag2, lag3, lag4, lag5, vol, tdy):
        self.lag1 = 0
        self.lag2 = 0
        self.lag3 = 0
        self.lag4 = 0
        self. lag5 = 0
        self.vol = 0
        self.tdy = 0
        self.path = 'D:/STUDY/Self Study/3.MachineLearning_and_DataScience/Python/x.Project/ml_app_knn_clsf/'
        self.data = pd.read_csv(self.path + "Smarket.csv")

        self.new_data = self.data.drop(columns=["Year", "Unnamed: 0"], axis=1)
        self.scale = StandardScaler()

        self.X = pd.DataFrame(self.scale.fit_transform(self.new_data.drop("Direction", axis=1),),
                              columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today'])
        self.y = self.new_data.Direction
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.3,
                                                                                random_state=42,
                                                                                stratify=self.y)



    def predict(self):

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(self.X_train.values, self.y_train)

        new_val = np.array([[self.lag1, self.lag2, self.lag3, self.lag4, self.lag5, self.vol, self.tdy]])
        #print(f'new_val shape : {new_val.shape}')

        prediction = knn.predict(new_val)

        print(''.join(prediction))


# exe = MarketKNN()
#
# exe.predict()