#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


sample_submission.head()


# In[6]:


train.columns


# In[7]:


test.columns


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


train.info()


# In[11]:


test.info()


# In[12]:


train_id = train.pop('user_id')
test_id =test.pop('user_id')


# In[13]:


numeric_feature = [feature for feature in train if train[feature].dtype in ['int64', 'float64']]
categorical_feature = [feature for feature in train if train[feature].dtype == 'object']


# In[14]:


numeric_feature


# In[15]:


categorical_feature


# In[16]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for cat in categorical_feature:
    train[cat] = le.fit_transform(train[cat])
    test[cat] = le.fit_transform(test[cat])


# low = 1,
# middle = 2,
# high 0
# 
# premium = 1,
# basic = 0

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[18]:


plt.figure(figsize=(1,1))
sns.countplot(data=train, x='target')


# In[19]:


sns.heatmap(train.corr(), linewidths=0.2, cmap='bwr' )


# In[20]:


target_corr = train.corr()['target'].drop('target')
target_corr = target_corr.apply(lambda x: abs(x))


# In[21]:


target_corr


# In[27]:


train[numeric_feature].hist(alpha=0.7, color='blue', edgecolor='k', layout=(4, 3), figsize=(10, 12))
plt.suptitle('Value', fontsize=16)
plt.xlabel('Value')
plt.ylabel('frequency')

# 그래프 간 간격 조절
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[28]:


def compare_categorical_data(df, categorical_col):    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=categorical_col, data=df)

    plt.title(categorical_col)
    plt.xlabel(categorical_col)
    plt.ylabel('Count')

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    fontsize=12,
                    color='black',
                    xytext=(0, 5),
                    textcoords='offset points'
        )

    plt.show()
    
compare_categorical_data(train, 'preferred_difficulty_level')
compare_categorical_data(train, 'subscription_type')
compare_categorical_data(train, 'payment_pattern')


# In[44]:


def draw_chart(df, col, bins=None):
    df0 = df[df['target'] == 0][col]
    df1 = df[df['target'] == 1][col].sample(3801)

    xmin = min(df0.min(), df1.min())
    xmax = max(df0.max(), df1.max())

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].hist(df0,  edgecolor='black', range=(xmin, xmax), label='Target 0')
    axs[0].hist(df1,  edgecolor='black', range=(xmin, xmax), label='Target 1')
    axs[0].set_title(f'Histogram of {col} by Target')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    sns.boxplot(x='target', y=col, data=df, ax=axs[1])
    axs[1].set_title(f'Boxplot of {col} by Target')
    axs[1].set_xlabel('Target')
    axs[1].set_ylabel('Value')

    describes = pd.concat([df0.describe(), df1.describe()], axis=1)
    describes.columns = ['Target 0', 'Target 1']

for col in numeric_feature:
    draw_chart(train, col)


# In[41]:


for i, col in enumerate(numeric_feature):
    plt.subplot(3,4,i+1)
    plt.title(col, fontsize=8)
    plt.boxplot(train[col])
plt.tight_layout()


# In[ ]:




