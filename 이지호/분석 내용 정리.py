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
import numpy as np

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
# 4) Column 설명

# - user_id: 사용자의 고유 식별자
# - subscription_duration: 사용자가 서비스에 가입한 기간 (월)
# - recent_login_time: 사용자가 마지막으로 로그인한 시간 (일)
# - average_login_time: 사용자의 일반적인 로그인 시간
# - average_time_per_learning_session: 각 학습 세션에 소요된 평균 시간 (분)
# - monthly_active_learning_days: 월간 활동적인 학습 일수
# - total_completed_courses: 완료한 총 코스 수
# - recent_learning_achievement: 최근 학습 성취도
# - abandoned_learning_sessions: 중단된 학습 세션 수
# - community_engagement_level: 커뮤니티 참여도
# - preferred_difficulty_level: 선호하는 난이도
# - subscription_type: 구독 유형
# - customer_inquiry_history: 고객 문의 이력
# - payment_pattern: 사용자의 지난 3개월 간의 결제 패턴을 10진수로 표현한 값.
#     - 7: 3개월 모두 결제함
#     - 6: 첫 2개월은 결제했으나 마지막 달에는 결제하지 않음
#     - 5: 첫 달과 마지막 달에 결제함
#     - 4: 첫 달에만 결제함
#     - 3: 마지막 2개월에 결제함
#     - 2: 가운데 달에만 결제함
#     - 1: 마지막 달에만 결제함
#     - 0: 3개월 동안 결제하지 않음
# - target: 사용자가 다음 달에도 구독을 계속할지 (1) 또는 취소할지 (0)를 표시


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
train['subscription_type'] = train['subscription_type'].map({'Basic':0, 'Premium':1})

# test 라벨인코딩
test['preferred_difficulty_level'] = test['preferred_difficulty_level'].map({'Low':1,'Medium':2,'High':3})
test['subscription_type'] = test['subscription_type'].map({'Basic':0, 'Premium':1})

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
    


# In[24]:


# 2. 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
for col in numerical_cols:
    train[col] = scaler.fit_transform(train[col].values.reshape(-1,1))
    test[col] = scaler.transform(test[col].values.reshape(-1,1))


# In[25]:


train


# In[26]:


test


# In[27]:


# 3. 검증데이터 분리


# In[28]:


# 3-1. 구독이탈예측 검증데이터 분리

from sklearn.model_selection import train_test_split

x = train.drop('target', axis=1)
y = train['target']
test_x = train.drop('target', axis=1)
test_y = train['target']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=12)


# In[29]:


# 3-2. 구독유형추천 검증데이터 분리

X = train.drop(['target', 'subscription_type'], axis=1)
Y = train['subscription_type']
test_X = test.drop('subscription_type', axis=1)
test_Y = test['subscription_type']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, random_state=12)


# ## - 분석 모델링

# In[30]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import optuna
from optuna.integration import OptunaSearchCV


# ### 1. DT

# In[31]:


from sklearn.tree import DecisionTreeClassifier

def model_dt(x_train, x_val, y_train, y_val):
    
    model = DecisionTreeClassifier(random_state=12)
    
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, pred)
    
    return accuracy

best_dt = {'random_state': 12}
Best_dt = {'random_state': 12}


# ### 2. KNN

# In[32]:


from sklearn.neighbors import KNeighborsClassifier

score_list = []
def model_knn(trial, x_train, x_val, y_train, y_val):
        param_grid = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric',['euclidean', 'manhattan']),
        'p': trial.suggest_int('p', 1, 5)
        }
        model = KNeighborsClassifier(**param_grid)

        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

        for t, v in fold.split(x, y):
            x_train, x_val = x.iloc[t], x.iloc[v]
            y_train, y_val = y.iloc[t], y.iloc[v]

            model.fit(x_train, y_train)

            pred = model.predict(x_val)
            accuracy = accuracy_score(y_val, pred)

        return accuracy

# 구독 해지 예측 최적 파라미터
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: model_knn(trial, x_train, x_val, y_train, y_val), n_trials=10, n_jobs=-1)
best_knn = study.best_trial.params


# 구독 유형 추천 최적 파라미터
Study = optuna.create_study(direction='maximize')
Study.optimize(lambda trial: model_knn(trial, X_train, X_val, Y_train, Y_val), n_trials=10, n_jobs=-1)
Best_knn = Study.best_trial.params


print(f'KNN - 구독 해지 예측 최적 파라미터:{study.best_trial.params}')
print(f'KNN - 구독 해지 예측 최적 파라미터의 정확도:{study.best_trial.values[0]}')

print(f'KNN - 구독 유형 추천 최적 파라미터:{Study.best_trial.params}')
print(f'KNN - 구독 유형 추천 최적 파라미터의 정확도:{Study.best_trial.values[0]}')


# ### 3. XGboost

# In[33]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def model_xgb(x,y):
    model = xgb.XGBClassifier()
    param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'subsample': [0.3, 0.5, 0.55, 0.6, 0.7],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'n_estimators': [104, 109, 210, 300, 350, 500]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x, y)
    
    return [grid_search.best_params_, grid_search.best_score_]


# In[34]:


# XGboost fitting은 시간이 오래 걸렸으므로 결과만 따로 불러와 사용하도록 함.

best_xgb = {
    'colsample_bytree':0.5,
    'n_estimators':104,
    'max_depth':1,
    'min_child_weight':10,
    'random_state':12,
    'subsample':0.7,
    'n_jobs':-1,
    }

Best_xgb = {
    'colsample_bytree':0.5,
    'n_estimators':109,
    'max_depth':1,
    'min_child_weight':3,
    'random_state':12,
    'subsample':0.6,
    'n_jobs':-1,
    }


print("XGboost - 구독 해지 예측 최적 파라미터:{{'colsample_bytree': 0.5, 'max_depth': 1, 'min_child_weight': 10, 'n_estimators': 104, 'subsample': 0.7}}")
print("XGboost - 구독 해지 예측 최적 파라미터의 정확도:0.6156")

print("XGboost - 구독 유형 추천 최적 파라미터:{{'colsample_bytree': 0.5, 'max_depth': 1, 'min_child_weight': 3, 'n_estimators': 109, 'subsample': 0.6}}")
print("XGboost - 구독 유형 추천 최적 파라미터의 정확도:0.7695")


# ### 4. LightGBM

# In[35]:


from lightgbm import LGBMClassifier

def model_lgbm(trial, x_train, x_val, y_train, y_val):
    
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
        random_state=12
    )
    model.fit(x_train, y_train)

    # 검증 세트에서의 성능 평가
    pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, pred)
    return accuracy

# 구독 해지 예측 최적 파라미터
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: model_lgbm(trial, x_train, x_val, y_train, y_val), n_trials=10, n_jobs=-1)
best_lgbm = study.best_trial.params
best_lgbm['silent']=True

# 구독 유형 추천 최적 파라미터
Study = optuna.create_study(direction='maximize')
Study.optimize(lambda trial: model_lgbm(trial, X_train, X_val, Y_train, Y_val), n_trials=10, n_jobs=-1)
Best_lgbm = Study.best_trial.params
Best_lgbm['silent']=True

print(f'LightGBM - 구독 해지 예측 최적 파라미터:{study.best_trial.params}')
print(f'LightGBM - 구독 해지 예측 최적 파라미터의 정확도:{study.best_trial.values[0]}')

print(f'LightGBM - 구독 유형 추천 최적 파라미터:{Study.best_trial.params}')
print(f'LightGBM - 구독 유형 추천 최적 파라미터의 정확도:{Study.best_trial.values[0]}')


# ## - 분석

# ### 0. 예측 결과 확인 함수

# In[36]:


best_params_list = [best_dt, best_knn, best_xgb, best_lgbm]
Best_params_list = [Best_dt, Best_knn, Best_xgb, Best_lgbm]
models_label = ['DT', 'KNN', 'XGBoost', 'LightGBM']
models = [DecisionTreeClassifier,KNeighborsClassifier,XGBClassifier,LGBMClassifier]

def filter_params(model_class, params):
    valid_params = model_class().get_params().keys()
    return {k: v for k, v in params.items() if k in valid_params}


# ### 1. 구독 해지 예측

# In[37]:


for i, param in enumerate(best_params_list):
    model = models[i](**filter_params(models[i], best_params_list[i]))
    model.fit(x,y)
    pred = model.predict(test_x)

    print(f'-----------------------{models_label[i]}-------------------------')
    print(classification_report(test_y, pred))
    
    # 혼동행렬 시각화
    cf_matrix = confusion_matrix(test_y, pred)
   
    group_names = ['TN','FP', 'FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    plt.title(models_label[i])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    plt.show()


# ### 2. 구독 유형 추천

# In[38]:


for i, param in enumerate(best_params_list):
    model = models[i](**filter_params(models[i], Best_params_list[i]))
    model.fit(X,Y)
    pred = model.predict(test_X)
    
    print(f'-----------------------{models_label[i]}-------------------------')
    print(classification_report(test_Y, pred))
    
    # 혼동행렬 시각화
    cf_matrix = confusion_matrix(test_Y, pred)
   
    group_names = ['TN','FP', 'FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    plt.title(models_label[i])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    plt.show()


# ## - 결론
