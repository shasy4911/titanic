
# coding: utf-8

# ## Titanic Shipwreck Prediction

# ## 1. Importing the libraries

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## 2.Collecting the data

# In[2]:


#Importing the dataset
train_data = pd.read_csv('titanic_train.csv')


# In[3]:


test_data = pd.read_csv('titanic_test.csv')


# ## 3. EDA

# In[4]:


train_data.info()
print('----------------------------------------')
test_data.info()


# In[5]:


train_data.head()


# In[6]:


test_data.head()


# ## Cleaning the training data

# ## 4. Feature Engineering
# 
# **It means using the domain knowledge of the data for the feature to mould the features such that it works easily and meaningfully with the machine learning models.**

# ### 4.1. Name

# In[7]:


train_data['Name'].head()


# In[8]:


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Invalid'
    
def change_title(title):
    if title in ['Mr']:
        return 0
    elif title in ['Master']:
        return 1
    elif title in ['Ms','Mlle','Miss','Mme','Mrs']:
        return 2
    else:
        return 3


# In[9]:


train_data['Title'] = train_data['Name'].apply(get_title).apply(change_title)


# In[10]:


train_data['Title'].value_counts()


# In[11]:


test_data['Title'] = test_data['Name'].apply(get_title).apply(change_title)


# In[12]:


test_data['Title'].value_counts()


# In[13]:


train_data.head()


# In[14]:


train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)


# In[15]:


train_data.head()


# In[16]:


test_data.head()


# ### 4.2 Sex

# In[17]:


train_test_set = [train_data, test_data]


# In[18]:


# Encoding the categorical column and creating the dummy variable
#sex = pd.get_dummies(train_data['Sex'], drop_first=True)
#sex = pd.get_dummies(test_data['Sex'], drop_first=True)
#train_data = pd.concat([train_data, sex], axis=1)
#test_data = pd.concat([test_data, sex], axis=1)
sex = {'male':0, 'female':1}
for data in train_test_set:
    data['Sex'] = data['Sex'].map(sex)


# In[19]:


train_data.head()


# ### 4.3. Age

# In[20]:


#Impute the missing values
train_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('mean'), inplace=True)
test_data['Age'].fillna(test_data.groupby('Title')['Age'].transform('mean'), inplace=True)


# In[21]:


# For changing the Age to Categorical variable
for data in train_test_set:
    data.loc[data['Age'] <= 15, 'Age'] = 0
    data.loc[(data['Age'] > 15) & (data['Age'] <= 25), 'Age'] = 1
    data.loc[(data['Age'] > 25) & (data['Age'] <= 35), 'Age'] = 2
    data.loc[(data['Age'] > 35) & (data['Age'] <= 55), 'Age'] = 3
    data.loc[data['Age'] > 55, 'Age'] = 4


# In[22]:


train_data.head()


# In[23]:


#Impute the missing values
#test_data['Age'].fillna(test_data.Age.mean(), inplace=True)


# In[ ]:


#Converting the Age to int from float
#train_data['Age'] = train_data['Age'].astype(int)
#test_data['Age'] = test_data['Age'].astype(int)


# ### 4.4. Embarked

# In[23]:


for data in train_test_set:
    data['Embarked'] = data['Embarked'].fillna('S')


# In[24]:


embark = {'S':0, 'C':1, 'Q':2}
for data in train_test_set:
    data['Embarked'] = data['Embarked'].map(embark)


# In[25]:


train_data.head()


# ### 4.5. Fare

# In[26]:


# Fit missing values of Fare with mean fare of each Pclass
train_data['Fare'].fillna(train_data.groupby('Pclass')['Fare'].transform('mean'), inplace=True)
test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform('mean'), inplace=True)


# In[27]:


# For changing the Fare to Categorical variable
for data in train_test_set:
    data.loc[data['Fare'] <= 17, 'Fare'] = 0
    data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30), 'Fare'] = 1
    data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100), 'Fare'] = 2
    data.loc[data['Fare'] > 100, 'Fare'] = 3


# In[29]:


train_data.head()


# In[28]:


train_data['Fare'].head()


# ### 4.6. Cabin

# In[29]:


for data in train_test_set:
    data['Cabin'] = data['Cabin'].str[:1]


# In[30]:


train_data['Cabin'].value_counts()


# In[31]:


cabin_map = {'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F':2.0, 'G':2.4, 'T':2.8}
# The range is given as such so that that the model should perform correctly
for data in train_test_set:
    data['Cabin'] = data['Cabin'].map(cabin_map)


# In[32]:


# Fit missing values of Cabin with mean fare of each Pclass
train_data['Cabin'].fillna(train_data.groupby('Pclass')['Cabin'].transform('mean'), inplace=True)
test_data['Cabin'].fillna(test_data.groupby('Pclass')['Cabin'].transform('mean'), inplace=True)


# In[33]:


train_data['Cabin'].head()


# ### 4.7. Family Size

# In[34]:


train_data['Family'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['Family'] = test_data['SibSp'] + test_data['Parch'] + 1


# In[35]:


train_data['Family'].value_counts()


# In[36]:


test_data['Family'].value_counts()


# In[37]:


family_map = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4.0}
for data in train_test_set:
    data['Family'] = data['Family'].map(family_map)


# In[38]:


train_data.head()


# In[39]:


test_data.head()


# In[40]:


train_data.drop(['PassengerId','Sex', 'SibSp','Parch','Ticket'], inplace=True, axis=1)
test_data.drop(['Sex','SibSp','Parch', 'Ticket'], inplace=True, axis=1)


# In[41]:


# Now categorizing the dataset into independent and dependent variables
X = train_data.drop('Survived', axis=1) # This is independent variable
y = train_data['Survived']


# In[42]:


X.shape, y.shape


# In[43]:


X.head()


# In[46]:


'''
# Scaling the values in the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#X_test = scaler.transform(X_test)

#y_train = scaler.fit_transform(y_train)
'''


# In[44]:


X


# In[ ]:


#Evaluating the model
#from sklearn.metrics import confusion_matrix, classification_report
#print(confusion_matrix(y_test, pred))
#print('--------------------------------------------')
#print(classification_report(y_test, pred))


# ## 5. Modelling the dataset
# **Checking the accuracy with different classifiers**

# In[45]:


# importing the machine learning models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ### Cross Validation (K-fold)

# In[46]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### 6.1 Logistic Regression

# In[47]:


# Predicting with Logistic Regression
log_reg = LogisticRegression()

#log_reg.fit(X_train, y_train)

score = cross_val_score(log_reg, X, y, cv = k_fold, scoring='accuracy')
print(score)
#pred = log_reg.predict(X_test)

#Evaluating the accuracy on the training and test data
#print('The accuracy of the algorithm on the training data is {}'.format(log_reg.score(X_train, Y_train)))

#print('The accuracy of the algorithm on the test data is {}'.format(log_reg.score(y_test, pred)))


# In[48]:


#Accuracy from Logistic Regression
round(np.mean(score)*100, 2)


# ### 6.2 kNN

# In[49]:


knn = KNeighborsClassifier(n_neighbors=13)
score = cross_val_score(knn, X, y, cv=k_fold, scoring='accuracy')
print(score)


# In[50]:


#Accuracy of KNN from KNN Score
round(np.mean(score)*100, 2)


# ### 6.3 Decision Tree

# In[51]:


tree = DecisionTreeClassifier()
score = cross_val_score(tree, X, y, cv=k_fold, scoring='accuracy')
print(score)


# In[52]:


#Accuracy from Decision Tree
round(np.mean(score)*100, 2)


# ## 6.4 Random Forest

# In[53]:


forest = RandomForestClassifier(n_estimators=13)
score = cross_val_score(forest, X, y, cv=k_fold, scoring='accuracy')
print(score)


# In[54]:


#Accuracy from the Random Forest
round(np.mean(score)*100, 2)


# ## 6.5 Naive Bayes

# In[55]:


nb = GaussianNB()
score = cross_val_score(forest, X, y, cv=k_fold, scoring='accuracy')
print(score)


# In[56]:


#Accuracy from the Random Forest
round(np.mean(score)*100, 2)


# ## 6.6 Support Vector Machine

# In[57]:


svc = SVC()
score = cross_val_score(svc, X, y, cv=k_fold, scoring='accuracy')
print(score)


# In[58]:


#Accuracy from SVM
round(np.mean(score)*100, 2)


# ## 7. Testing

# In[62]:


svc = SVC()
svc.fit(X, y)

test = test_data.drop('PassengerId', axis=1)
prediction = svc.predict(test)


# In[63]:


submission = pd.DataFrame({
                    'PassengerId': test_data['PassengerId'],
                    'Survived': prediction
            })
submission.to_csv('predicted.csv', index=False)


# In[64]:


submission = pd.read_csv('predicted.csv')
submission

