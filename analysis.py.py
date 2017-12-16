
# coding: utf-8

# ## Titanic Survival Analysis

# ## 1. Importing the libraries

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## 2. Collecting the data

# In[2]:


#Importing the dataset
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')


# ### 3. Exploratory Data Analysis(EDA)

# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.shape


# In[6]:


test_data.shape


# In[7]:


train_data.info()
print('----------------------------------')
test_data.info()


# In[8]:


train_data.describe()


# In[9]:


test_data.describe()


# In[10]:


train_data.isnull().sum()


# In[11]:


test_data.isnull().sum()


# In[12]:


train_data.columns
# df.axes  # The axes can also be used for the same


# In[13]:


#Plot for checking the missing values from training set
sns.heatmap(train_data.isnull(), yticklabels=False,cbar=False, cmap='viridis')


# In[14]:


#Plot for checking the missing values from test set
sns.heatmap(test_data.isnull(), yticklabels=False,cbar=False, cmap='viridis')


# **The heatmap visualization clearly shows that there are a fair amount of data missing in Age column and Cabin column**

# In[15]:


# Style of the seaborn for the plots 
sns.set_style('whitegrid')


# In[16]:


# Checking the Survival ratio and also of different Sex
sns.countplot(x='Survived', data=train_data, hue='Sex', palette='viridis')

plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)


# **Here 0 means not Survived and 1 means Survived**

# In[17]:


# Checking the Survival ratio and also of different classes of people
sns.countplot(x='Survived', data=train_data, hue='Pclass', palette='viridis')

plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)


# In[18]:


#Checking the age group of the people onboard
sns.distplot(train_data['Age'].dropna(), bins=30, kde=False, color='blue')


# In[19]:


#Checking the age group of the people onboard with built-in plot of pandas
train_data['Age'].plot.hist(bins=30)


# In[20]:


# Checking the Sibilings and Spouses onboard
sns.countplot(x='SibSp', data=train_data)


# In[21]:


plt.figure(figsize=(10,5))
train_data['Fare'].plot(kind='hist', bins=30)
plt.xlabel('Fares')


# ** By looking above plot of Fares, it clearly seems that mostly it is distributed towards cheaper fare tickets.
#    It also makes sense because most people were of Third Class who bought the tickets. **

# In[22]:


#Checking the Survival of male vs female
# 1 means Survived and 0 means not Survived.
fig, axes = plt.subplots(1,2)

train_data[train_data['Sex']=='male'].Survived.value_counts().plot(kind='bar', ax=axes[0], title='Male Survivor')
train_data[train_data['Sex']=='female'].Survived.value_counts().plot(kind='bar', ax=axes[1], title='Female Survivor')

