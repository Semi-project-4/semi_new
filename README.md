# 구독 해지 예측 및 구독 유형 추천 모델 구현


</details>

## 1. Intro
### - 목적
해당 프로젝트는 교육 수강생들의 '구독 기간', '로그인 활동', '학습 세션 참여도'와 같은 데이터 분석을 통해 이들이 서비스 구독을 계속할지 예측하고, 구독 유형을 추천하는 AI 알고리즘을 개발하는 것을 목표로 한다. 프로젝트는 4인으로 구성된 팀을 이루어 협업을 경험하고, 각각 다른 머신러닝 알고리즘을 사용하여 이를 공유하여 insight를 넓히는 방향으로 진행한다. 

### - 팀 소개
![팀소개](/image/intro_team.JPG) 

### - 프로젝트 의의
[맥킨지](https://opentutorials.org/module/782/6083)의 자료에 따르면 2025년 디지털플랫폼 전체 매출은 약 60조 달러, 원화로 약 7경 2000조원으로 추정되며 글로벌 전체 기업 매출 30%를 플랫폼 비즈니스가 담당할 것이라고 예상한다. 디지털 기술 기반 플랫폼 비즈니스 위주의 변화가 계속되는 가운데, 교육시장에서 고객 데이터 활용한 에듀테크 키워드가 활성화 되고있다. 

교육 플랫폼 경우, 고객 니즈에 맞는 콘텐츠 제공과 이탈 조짐이 보이는 고객을 사전 식별하여 수익을 유지해야 한다. 해당 프로젝트는 플랫폼 수익 유지에 기여하는 모델 개발의 시작점이라는 의의를 가진다.

### - 진행 일정
![진행일정](/image/schedule.JPG) 

### - 활용 도구
![진행일정](/image/tools.JPG) 

## 2. 데이터 수집·처리

### - 데이터 수집
DACON에서 주관한 ['해커톤 37회 학습 플랫폼 이용자 구독 갱신 예측 대회'](https://dacon.io/competitions/official/236179/data)에서 배포한 학습 플랫폼 이용자 데이터셋 csv파일을 사용하였다.

### - 데이터 구분

<details>
<summary>Column</summary>
<div markdown="1">

- user_id: 사용자의 고유 식별자  
- subscription_duration: 사용자가 서비스에 가입한 기간 (월) 
- recent_login_time: 사용자가 마지막으로 로그인한 시간 (일)  
- average_login_time: 사용자의 일반적인 로그인 시간  
- average_time_per_learning_session: 각 학습 세션에 소요된 평균 시간 (분)
- monthly_active_learning_days: 월간 활동적인 학습 일수
- total_completed_courses: 완료한 총 코스 수
- recent_learning_achievement: 최근 학습 성취도
- abandoned_learning_sessions: 중단된 학습 세션 수
- community_engagement_level: 커뮤니티 참여도
- preferred_difficulty_level: 선호하는 난이도
- subscription_type: 구독 유형
- customer_inquiry_history: 고객 문의 이력
- payment_pattern: 사용자의 지난 3개월 간의 결제 패턴을 10진수로 표현한 값.  
    7: 3개월 모두 결제함  
    6: 첫 2개월은 결제했으나 마지막 달에는 결제하지 않음  
    5: 첫 달과 마지막 달에 결제함  
    4: 첫 달에만 결제함  
    3: 마지막 2개월에 결제함  
    2: 가운데 달에만 결제함  
    1: 마지막 달에만 결제함  
    0: 3개월 동안 결제하지 않음  
- target: 사용자가 다음 달에도 구독을 계속할지 (1) 또는 취소할지 (0)를 표시
</div>
</details>  

### - 데이터 인코딩



## 3. 데이터 분석

### - EDA

### - 데이터 전처리

### - 모델링

### 모델링 평가

## 4. 결론

### - 프로젝트 개선점

### - 프로젝트 평가





