#!/usr/bin/env python
# coding: utf-8

# # 교육 과정 현황 데이터를 활용한 구독 예측 및 구독 유형 추천 서비스 모델 구현
# 
# ## - 개요
#  해당 프로젝트는 교육 수강생들의 '구독 기간', '로그인 활동', '학습 세션 참여도'와 같은 데이터 분석을 통해 이들이 서비스 구독을 계속할지 예측하고, 구독 유형을 추천하는 AI 알고리즘을 개발하는 것을 목표로 한다. 프로젝트는 4인으로 구성된 팀을 이루어 협업을 경험하고, 각각 다른 머신러닝 알고리즘을 사용하여 이를 공유하여 insight를 넓히는 방향으로 진행한다. 
#  
# ## - 프로젝트 과정
# 
# 1. team
#     1. 팀원 소개
#     2. 진행 단계
#     3. 사용 tool
# 
# 2. EDA
# 
# 3. 데이터 전처리
# 
# 4. 모델링
#     1. DT
#     2. KNN
#     3. XGboost
#     4. LightGBM
# 
# 5. 평가 

# ### 1. 데이터 불러오기

# In[2]:


import pandas as pd

train = pd.read_csv('../../데이터/train.csv')
test = pd.read_csv('../../데이터/test.csv')


# ### 2. 데이터 확인

# In[3]:


train.head()


# In[4]:


test.head()


# ### 3. 수치형 데이터와 범주형 데이터로 나누기

# In[5]:


train.info()


# In[6]:


numeric_feature = ['subscription_duration', 'recent_login_time',
       'average_login_time', 'average_time_per_learning_session',
       'monthly_active_learning_days', 'total_completed_courses',
       'recent_learning_achievement', 'abandoned_learning_sessions',
       'community_engagement_level', 'customer_inquiry_history',]
categorical_feature = ['preferred_difficulty_level','subscription_type','payment_pattern']


# In[7]:


train[numeric_feature].describe()


# In[8]:


for col in categorical_feature:
    display(train[col].value_counts())


# ### 4. 데이터 시각화

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# #### 4-1. Target 데이터 갯수

# In[10]:


target_count = train['target'].value_counts().sort_index()
plt.bar(target_count.index, target_count.values)


for i, count in enumerate(target_count.values):
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()


# #### 4-2. 수치형 데이터 분포

# In[11]:


numeric_train = train[numeric_feature]
numeric_train.hist(alpha=0.5, edgecolor='k', layout=(6,2), figsize=(6, 12))

plt.suptitle('Value', fontsize=16)
plt.xlabel('Value')
plt.ylabel('frequency')

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()


# In[12]:


plt.figure(figsize=(10,8))
corr_train = train[numeric_feature].corr()
sns.heatmap(corr_train, annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()


# #### 4-3. 범주형 변수 분포

# In[13]:


def compare_categorical_data(df, col):    
    plt.figure(figsize=(5, 5))
    ax = sns.countplot(x=col, data=df)

    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('count')

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black',
                    xytext=(0, 5),
                    textcoords='offset points'
        )

    plt.show()

for col in categorical_feature:
    compare_categorical_data(train, col)


# ### 5. 데이터 전처리

# #### 5-1. 이상치 제거

# In[14]:


def replace_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df

for col in numeric_feature:
    train = replace_outliers(train, col)

train.reset_index(drop=True, inplace=True)


# #### 5-2. 라벨 인코딩 

# In[15]:


def transfer_level_data(x):
    if x == 'High':
        return 3
    elif x == 'Medium':
        return 2
    else:
        return 1

train['preferred_difficulty_level'] = train['preferred_difficulty_level'].apply(transfer_level_data)
test['preferred_difficulty_level'] = test['preferred_difficulty_level'].apply(transfer_level_data)


# In[16]:


train['subscription_type'] = train['subscription_type'].apply(lambda x: 1 if x == 'Premium' else 0)
test['subscription_type'] = test['subscription_type'].apply(lambda x: 1 if x == 'Premium' else 0)


# #### 5-3. 원핫인코딩

# In[17]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(train[['payment_pattern']])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['payment_pattern']))

train = train.drop(columns='payment_pattern')
train = pd.concat([train, encoded_df], axis=1)

encoded = encoder.transform(test[['payment_pattern']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['payment_pattern']))

test = test.drop(columns='payment_pattern')
test = pd.concat([test, encoded_df], axis=1)


# #### 5-4. 스케일링

# In[18]:


from sklearn.preprocessing import MinMaxScaler

for col in numeric_feature:
    scaler = MinMaxScaler()
    train[col] = scaler.fit_transform(train[col].values.reshape(-1,1))
    test[col] = scaler.transform(test[col].values.reshape(-1,1))


# ### 6. 모델링

# In[19]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

x = train.drop(columns=['user_id','target'])
y = train['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)

model =lgb.LGBMClassifier(n_estimators=50, force_row_wise=True,learning_rate=0.01, verbose=0)
model.fit(x_train, y_train)
pred = model.predict(x_test)


# In[20]:


result = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

result.loc['LightGBM'] = [accuracy_score(y_test, pred), precision_score(y_test, pred), 
                 recall_score(y_test, pred), f1_score(y_test, pred)]

result


# In[25]:


import optuna
from lightgbm import LGBMClassifier

def objective(trial, X_train, X_val, y_train, y_val):
    
    # optuna를 이용한 하이퍼 파라미터 조정
    num_leaves = trial.suggest_int('num_leaves', 20, 3000, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    

    # 모델 생성 및 훈련
    model = LGBMClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        num_leaves=num_leaves,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 검증 세트에서의 성능 평가
    pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, pred)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, x_train, x_test, y_train, y_test), n_trials=10, n_jobs=-1)

result_score = study.best_trial.values[0]
result_score

