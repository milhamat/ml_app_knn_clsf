import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

path = 'D:/STUDY/Self Study/3.MachineLearning_and_DataScience/Python/x.Project/ml_app_knn_clsf/'
data = pd.read_csv(path + "Smarket.csv")


# def chg_dtyp(df):
#     """
#     df : is the feature or column from the y target
#     this helper help change the data type
#     from object(string) into integer
#     0 for Down
#     1 for Up
#     """
#     list = []
#     for data in df:
#         if data == 'Up':
#             list.append(1)
#         else:
#             list.append(0)
#         return list


# data_list = chg_dtyp(data.Direction)
# data['Outcome'] = data_list
# new_data = data.drop(columns=["Year", "Direction", "Unnamed: 0"], axis=1)
new_data = data.drop(columns=["Year", "Unnamed: 0"], axis=1)

scale = StandardScaler()
X = pd.DataFrame(scale.fit_transform(new_data.drop("Direction", axis=1),),
                 columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today'])
y = new_data.Direction

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

Lag1 = 0.28
Lag2 = -0.60
Lag3 = -2.90
Lag4 = 2.00
Lag5 = 5.79
Volume = 1.59
Today = 2.75

new_val = np.array([[Lag1, Lag2, Lag3, Lag4, Lag5, Volume, Today]])
#print(f'new_val shape : {new_val.shape}')

prediction = knn.predict(new_val)

print(prediction)