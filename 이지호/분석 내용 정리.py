#!/usr/bin/env python
# coding: utf-8

# # 교육 과정 현황 데이터를 활용한 구독 예측 및 구독 유형 추천 서비스 모델 구현

# ## 목차 

# ## - 개요
#  해당 프로젝트는 교육 수강생들의 '구독 기간', '로그인 활동', '학습 세션 참여도'와 같은 데이터 분석을 통해 이들이 서비스 구독을 계속할지 예측하고, 구독 유형을 추천하는 AI 알고리즘을 개발하는 것을 목표로 한다. 프로젝트는 4인으로 구성된 팀을 이루어 협업을 경험하고, 각각 다른 머신러닝 알고리즘을 사용하여 이를 공유하여 insight를 넓히는 방향으로 진행한다. 

# ### 1. 팀 소개

# ### 2. 주제 선정 배경

# ### 3. 진행 일정

# ### 4. 활용 tools

# ## - 데이터 수집 & 전처리

# ### 1. 과정

# ### 2. 데이터 수집

# In[1]:


# 1. 데이터 불러오기

import pandas as pd

train_origin = pd.read_csv('train.csv')
test_origin = pd.read_csv('test.csv')

# 2. 사용 데이터를 카피
train = train_origin.copy()
test = test_origin.copy()


# In[2]:


# 3. 불러온 정보 확인
train.info()
test.info()


# In[3]:


# info를 통해 확인한 정보
# 1) 데이터의 컬럼 갯수 (train은 15개, test는 14개)
# 2) 데이터 내 결측치가 존재하는지(없음)
# 3) column의 데이터 type이 무엇인지 (범주형과 수치형으로 구분 가능)


# ### 3. 데이터 전처리(1)

# In[4]:


# 1. 불러온 정보확인
train.head()


# In[5]:


# 2. 고유값 확인
train[train.columns].nunique()
test[test.columns].nunique()


# In[6]:


# 'user_id'는 고유값이므로 index를 대체할 수 있음.
# 이외에 'average_login_time', 'average_time_per_learning_session', 'recent_learning_achievement' column이
# 고유값을 가질 수 있으나, 이들은 우연히 같은 값을 가질 뿐 진정한 고유값이라 할 수 없음.


# In[7]:


train = train.set_index(['user_id'])
test = test.set_index(['user_id'])


# In[8]:


# 3. 수치형 데이터와 범주형 데이터로 분리

# info에서 알 수 있듯이 'preferred_difficulty_level','subscription_type' columns은 dtype이 object임
# payment_pattern'은  사용자의 지난 3개월 간의 결제 패턴을 10진수로 표현한 값이므로 범주형 데이터임.
# 따라서 위 세가지 columns를 범주형 데이터, 나머지 columns를 수치형 데이터로 구분 가능
# 단, 'payment_pattern' 의 경우 object 형식이 아니므로 따로 구분해둠 (이후 인코딩 방식에서 차이有)
# 'target'은 예측해야 하는 값이므로 제외시킴.

numerical_cols = [col for col in train.columns[:-2] if train[col].dtype in ['int64', 'float64'] and col != 'payment_pattern']
categorical_cols = [col for col in train.columns if train[col].dtype == 'object']


# In[9]:


numerical_cols


# In[10]:


categorical_cols


# In[11]:


# 4. 범주형 데이터 라벨인코딩


# In[12]:


# 4-1. 범주형 데이터 컬럼 확인

for col in categorical_cols:
    display(train[col].value_counts())


# In[13]:


# 'preferred_difficulty_level' : 고유값으로 Low, Medium, High를 가지고 비율이 약 5:3:2
# 'subscription_type' : 고유값으로 Basic, Premium을 가지고 비율이 약 6:4
# 'payment_pattern' : 고유값들의 비율에 큰 차이 없음. 


# In[14]:


# 4-1)을 시각화
import matplotlib.pyplot as plt
import seaborn as sns

for col in categorical_cols:
    ratio = train[col].value_counts()
    labels = train[col].unique()
    wedgeprops={'width': 1, 'edgecolor': 'w', 'linewidth': 1}
    
    plt.figure(figsize = (3,3))
    plt.title(col)
    plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, wedgeprops=wedgeprops)
    plt.show()
    


# In[15]:


# 4-2) 라벨 인코딩

# train 라벨인코딩
train['preferred_difficulty_level'] = train['preferred_difficulty_level'].map({'Low':1,'Medium':2,'High':3})
train['subscription_type'] = train['subscription_type'].map({'Basic':1, 'Premium':2})

# test 라벨인코딩
test['preferred_difficulty_level'] = test['preferred_difficulty_level'].map({'Low':1,'Medium':2,'High':3})
test['subscription_type'] = test['subscription_type'].map({'Basic':1, 'Premium':2})

# 확인
train[categorical_cols].head()


# In[16]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(train[['payment_pattern']])

encoded_df = pd.DataFrame(encoded, index=train.index, columns=encoder.get_feature_names_out(['payment_pattern']))

train = train.drop(columns='payment_pattern')
train = pd.concat([train, encoded_df], axis=1)

encoded = encoder.transform(test[['payment_pattern']])
encoded_df = pd.DataFrame(encoded, index=test.index, columns=encoder.get_feature_names_out(['payment_pattern']))

test = test.drop(columns='payment_pattern')
test = pd.concat([test, encoded_df], axis=1)


# ### 4. EDA

# In[17]:


# 1. 수치형 데이터 분포 확인
train[numerical_cols].hist(alpha=0.5, edgecolor='k', layout=(6,2), figsize=(6, 12))

plt.suptitle('Value', fontsize=16)
plt.xlabel('Value')
plt.ylabel('frequency')

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()


# In[18]:


# 1) 'subscription_duration','recent_login_time','monthly_active_learning_days' columns는 균일 분포
# 2) 'average_login_time', 'total_completed_courses', 'recent_learning_achievement' columns는 종형 분포
# 3) 'average_time_per_learning_session', 'abandoned_learning_sessions', 'customer_inquiry_history' columns는 왜도가 높은 분포
#     왜도가 높으므로 해당 columns에서는 이상치가 있을 것임을 알 수 있다.


# In[19]:


# 2. columns의 상관관계


# In[20]:


# 2-1. 수치형 columns의 상관관계 확인
corr_train = train[numerical_cols + ['target']].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_train, annot=True, fmt='.3f', cmap='coolwarm', annot_kws={'size': 8})
plt.title('[Heatmap] train_numerical_cols')

plt.tick_params(axis='both', which='both', labelsize=8)
plt.gca().get_xticklabels()[3].set_fontsize(14)
plt.gca().get_xticklabels()[3].set_color('red') 
plt.gca().get_yticklabels()[10].set_fontsize(14)  
plt.gca().get_yticklabels()[10].set_color('red')  


plt.show()


# In[21]:


# 1) 수치형 columns에서 'average_time_per_session' 과 'target'의 상관 관계가 있을 것으로 예상


# In[22]:


# 2-2. 범주형 columns의 상관관계 확인
corr_train = train[categorical_cols + ['target']].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_train, annot=True, fmt='.3f', cmap='coolwarm', annot_kws={'size': 16})
plt.title('[Heatmap] train_categorical_cols')
plt.tick_params(axis='both', which='both', labelsize=10) 

plt.show()


# ### 5. 데이터 전처리(2)

# In[23]:


# 1. 이상치 제거
def replace_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df

outlier_columns = ['average_time_per_learning_session', 'abandoned_learning_sessions', 'customer_inquiry_history']

for col in outlier_columns:
    train = replace_outliers(train, col)
    


# In[25]:


# 2. 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
for col in numerical_cols:
    train[col] = scaler.fit_transform(train[col].values.reshape(-1,1))
    test[col] = scaler.transform(test[col].values.reshape(-1,1))


# ## - 분석 모델링

# ### 1. DT

# ### 2. KNN

# ### 3. XGboost

# ### 4. LightGBM

# ## - 모델 평가

# ## - 결론
