#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set()
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("Smarket.csv")


# ## Data Exploration and Analysis

# In[3]:


# see the first 5 data in the dataset
data.head()


# In[4]:


data.shape
# so the data construct from 1250 observable data or rows 
# and 10 features columns


# In[5]:


# see the data type each column(features)
data.info()
# we can see that the column or feature number 9 has 
# data type as object or string but the problem is 
# in machine learning generaly the algorithm doesn't 
# work with non numerical value so we need to convert 
# the string into integer 0 for down and 1 for up
# we can say this method as one hot encoding


# In[6]:


# see the statistical data each columns(features)
data.describe()
# we can also see that the Direction can't be desribe 
# because it's not a numerical data


# In[7]:


# see if there any Null or NA value in our dataset
print(data.isnull().sum())
# from the result we can see there is no Null or NA value in the dataset 


# In[8]:


def change_obj2int(df):
    """
    df : is the feature or column from the y target 
    this helper help change the data type
    from object(string) into integer
    0 for Down
    1 for Up
    """
    list = []
    for data in df:
        if data == 'Up':
            list.append(1)
        else:
            list.append(0)
    return list
# so basicly the helper will return the data into
# list data structure so we need to convert and 
# add the new column to our dataset


# In[9]:


data_list = change_obj2int(data.Direction)


# In[10]:


#print(data_list)
# for checking the result


# In[11]:


# add our new list into the data frame
data['Outcome'] = data_list


# In[12]:


data.head()


# In[13]:


data.tail()
# we can see that our method perform well


# ## Data Visualization

# In[14]:


vis = data.hist(figsize=(20,20), color="skyblue")
# maybe we can ignore some of features like Unnamed 0 because the data
# is only number of the datas, Outcome because it's spose only have 
# two data variation also year


# In[15]:


#new_data = data.drop('Year', axis= 1) # if the object only one
new_data = data.drop(columns=['Year', 'Direction', 'Unnamed: 0'], axis= 1)


# In[16]:


new_data.head()


# ## Data Preprocessing

# In[17]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()


# In[18]:


X = pd.DataFrame(scale.fit_transform(new_data.drop("Outcome", axis=1),),
                 columns=['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 
                          'Volume', 'Today'])


# In[19]:


X.head()
# we can see the data has same or similar scale
# because we apply the stadardscaler 


# In[20]:


y = new_data.Outcome


# In[21]:


#y.head()
# only for checking


# ## Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state= 42, 
                                                    stratify=y)


# In[23]:


print('X_training : ',len(X_train))
print('y_training : ',len(y_train))
# size of the training


# In[24]:


print('X_testing : ',len(X_test))
print('y_testing : ',len(y_test))
# size of the test


# In[25]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


# ## Model Validation

# In[26]:


# score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100, list(map(lambda x: x+1, train_scores_ind))))


# In[27]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[28]:


knn = KNeighborsClassifier(7)
knn.fit(X_train, y_train)

print('Train Accuracy at k=7 : ',knn.score(X_train, y_train))
print('Test Accuracy at k=7  : ',knn.score(X_test, y_test))


# ### Result Visualisation

# In[29]:


plt.figure(figsize=(18,8))
p = sns.lineplot(range(1,15), train_scores, marker='*', label='Train Score')
p = sns.lineplot(range(1,15), test_scores, marker='o', label='Test Score')
# from 15 iteration we can see that at k=7 is the peak of the accuracy of test


# In[30]:


average_test = sum(test_scores) / len(test_scores)
average_test


# In[31]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(7)

knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# ### Confusion Matrix

# In[32]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[33]:


y_pred = knn.predict(X_test) # to predict we must use X_test or maybe our prefer data set

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred) # make confusion matrix
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="magma" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# ### ROC

# In[34]:


from sklearn.metrics import roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[35]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()


# ## Hyper Parameter Optimization
# ### GridSearch

# In[36]:


# applying the grid search method
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
#In case of classifier like knn the parameter to be tuned is n_neighbors

param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid, cv=5)
knn_cv.fit(X,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))
# for 46 neighbor is that not having some overfitting problem ???


# #### Experiment Prediction
# we can make our self value for each feature and test to our model is it will predict our new value with goods?

# In[37]:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)


# In[38]:


Lag1 = 0.28
Lag2 = -0.60
Lag3 = -2.90
Lag4 = 2.00
Lag5 = 5.79
Volume = 1.59
Today = 2.75

new_val = np.array([[Lag1, Lag2, Lag3, Lag4, Lag5, Volume, Today]])
print(f'new_val shape : {new_val.shape}')


# In[39]:


prediction = knn.predict(new_val)


# In[40]:


prediction


# In[41]:


if prediction == 1:
    print('Up')
else:
    print('Down')


# In[ ]:





# In[ ]:




