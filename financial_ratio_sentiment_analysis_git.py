#%%
import pandas as pd
df_bs = pd.read_table('/Users/seonjin/ColabProjects/2020_4Q_BS_20210327040036/2020_사업보고서_01_재무상태표_20210327.txt', encoding='cp949', thousands = ',') 
df_cf = pd.read_table('/Users/seonjin/ColabProjects/2020_4Q_CF_20210327040148/2020_사업보고서_04_현금흐름표_20210327.txt', encoding='cp949', thousands = ',') 
df_pl = pd.read_table('/Users/seonjin/ColabProjects/2020_4Q_PL_20210327040117/2020_사업보고서_03_포괄손익계산서_연결_20210327.txt', encoding='cp949', thousands = ',') 
# %%
df = pd.concat([df_bs,df_cf,df_pl], ignore_index=True)
# %%
df.info()
del df_bs,df_cf,df_pl
# %%
'''
# 건전 / 부도 동일 코드 
'''
df['항목명'] = df['항목명'].str.replace("[^a-zA-Zㄱ-힗]","")
df['종목코드'] = df['종목코드'].str.replace("[^0-9a-zA-Zㄱ-힗]","")
# %%
df['항목명'] = df['항목명'].str.replace("수익매출액","매출액") # 두개겹치지않는다
df['항목명'] = df['항목명'].str.replace("이익잉여금결손금","이익잉여금")
df['항목명'] = df['항목명'].str.replace("당기순이익손실","당기순이익") 
df['항목명'] = df['항목명'].str.replace("영업이익손실","영업이익") 
# %%
df.groupby(by='결산월').count() #12월꺼만 써도 될듯? 부도는 아닐수도
# %%
df2 = df[df['결산월']==12]
# %%
df2.groupby(by='결산월').count()
# %%
df32 = pd.pivot_table(df2, index = ['종목코드','회사명','결산기준일'], columns='항목명', values = '전기')
df31 = pd.pivot_table(df2, index = ['종목코드','회사명','결산기준일'], columns='항목명', values = '당기')
# %%
df4 = pd.merge(df31,df32, how='inner',on=['종목코드','회사명','결산기준일'], suffixes = ('','전기'))
# %%
df4.head()
# %%
# 상위 24개 변수
df4 = df4[['매출액',
'자본총계',
'이익잉여금',
'유동부채',
'유동자산',
'재무활동현금흐름',
'당기순이익',
'비유동자산',
'부채총계',
'비유동부채',
'현금및현금성자산',
'이자비용',
'영업이익',
'재고자산',
'자산총계',
'유형자산',
'매출액전기',
'자본총계전기',
'이익잉여금전기',
'유동부채전기',
'유동자산전기',
'재무활동현금흐름전기',
'당기순이익전기',
'비유동자산전기',
'부채총계전기',
'비유동부채전기',
'현금및현금성자산전기',
'이자비용전기',
'영업이익전기',
'재고자산전기',
'자산총계전기',
'유형자산전기']]
# %%
print(df4.shape)
print(df4.drop_duplicates().shape)
df4 = df4.drop_duplicates()
# %% 
df4.head()
# %%
df4.describe()
# %%
df4.info()
# %%
df4.isnull().sum()
# %%
# 재무비율 산식
try:
    df4['재고자산회전율'] =	df4['매출액'] / df4['재고자산']
except: pass
try:
    df4['자기자본비율'] =	df4['자본총계'] / df4['자산총계']
    df4['이익잉여금/총자산'] =	df4['이익잉여금'] / df4['자산총계']
    df4['유동부채/총자산'] =	df4['유동부채'] / df4['자산총계']
    df4['현금/총자산'] =	df4['현금및현금성자산'] / df4['자산총계']
    df4['유동자산/총자산'] =	df4['유동자산'] / df4['자산총계']
    df4['현금흐름/총자산'] =	df4['재무활동현금흐름'] / df4['자산총계']
    df4['총자산회전율'] =	df4['매출액'] / df4['자산총계']
    df4['총자산순이익율'] =	df4['당기순이익'] / (df4['자산총계'] + df4['자산총계전기']) / 2 
    df4['총부채/총자산'] =	df4['부채총계'] / df4['자산총계']
    df4['순운전자본/총자본']	= (df4['유동자산'] - df4['유동부채'])/df4['자산총계']
except: pass
try:
    df4['총자본순이익율'] =	df4['당기순이익'] / (df4['자본총계'] + df4['자본총계전기']) / 2
    df4['고정비율'] =	df4['비유동자산'] / df4['자본총계']
    df4['부채비율'] =	df4['부채총계'] / df4['자본총계']
    df4['유동부채비율'] =	df4['유동부채'] / df4['자본총계']
    df4['고정부채비율'] =	df4['비유동부채'] / df4['자본총계']
    df4['현금흐름/총자본'] =	df4['재무활동현금흐름'] / df4['자본총계']
    df4['총자본회전율'] =	df4['매출액'] / df4['자본총계']
except: pass
try:
    df4['현금비율'] =	df4['현금및현금성자산'] / df4['유동부채']
    df4['유동비율'] =	df4['유동자산'] / df4['유동부채']
    df4['이익잉여금/총부채']	= df4['이익잉여금'] / df4['부채총계']
except: pass
try:
    df4['고정자산회전율'] =	df4['매출액'] / df4['비유동자산']
except: pass
try:
    df4['영업활동이익/총부채'] =	df4['영업이익'] / df4['부채총계']
    df4['금융비용/총부채'] =	df4['이자비용'] / df4['부채총계']
    df4['유동부채/총부채'] =	df4['유동부채'] / df4['부채총계']
    df4['현금흐름/부채'] =	df4['재무활동현금흐름'] / df4['부채총계']
except: pass
try:
    df4['매출액순이익율'] =	df4['당기순이익'] / df4['매출액']
    df4['매출액영업이익율'] =	df4['영업이익'] / df4['매출액']
    df4['매출액이자비용율'] =	df4['이자비용'] / df4['매출액']
    df4['현금흐름/매출액'] =	df4['재무활동현금흐름'] / df4['매출액']
    df4['유동부채/매출액']	= df4['유동부채'] / df4['매출액']
except: pass
try:
    df4['이익잉여금/유동자산']	= df4['이익잉여금'] / df4['유동자산']
    df4['유동비율']	= df4['유동자산'] / df4['유동부채']
except: pass
try:
    df4['유형고정자산증가율'] =	df4['유형자산'] / df4['유형자산전기'] - 1
    df4['매출액증가율'] =	df4['매출액'] / df4['매출액전기'] - 1
    df4['순이익증가율'] =	df4['당기순이익'] / df4['당기순이익전기'] - 1
    df4['고정자산증가율'] =	df4['비유동자산'] / df4['비유동자산전기'] - 1
    df4['총자산증가율'] =	df4['자산총계'] / df4['자산총계전기'] - 1
except: pass
# %%
df4.info()
# %%
df4.describe()

# %%
dev = df4[[
'자산총계',
# '재고자산회전율',
'자기자본비율',
'이익잉여금/총자산',
'유동부채/총자산',
'현금/총자산',
'유동자산/총자산',
'현금흐름/총자산',
# '총자산회전율',
'총자산순이익율',
'총부채/총자산',
'순운전자본/총자본',
# '당기순이익/총자산', 
'총자본순이익율',
'고정비율',
# '고정장기적합율', 
'부채비율',
'유동부채비율',
'고정부채비율',
'현금흐름/총자본',
# '총자본회전율',
'현금비율',
'유동비율',
'이익잉여금/총부채',
# '고정자산회전율',
'영업활동이익/총부채',
# '금융비용/총부채',
'유동부채/총부채',
'현금흐름/부채',
# '매출액순이익율',
# '매출액영업이익율',
# '매출액이자비용율',
# '현금흐름/매출액',
# '유동부채/매출액',
'이익잉여금/유동자산',
'총자산증가율',
'유형고정자산증가율',
# '매출액증가율',
'순이익증가율',
'고정자산증가율'
]]
# %%
dev.info()
# %%
dev.isnull().sum()
# %%
dev.describe()
# %%
dev = dev.reset_index()
# %%
dev['종목코드'] = pd.to_numeric(dev['종목코드'])
# %%
##########################################################
'''
건전기업 부도여부 붙이기
'''
##########################################################
# %%
gb = pd.read_excel('data_company.xlsx', sheet_name = 'Sheet3', engine='openpyxl')
gb
# %%
gb_live = gb[gb['부도관리구분'] == 0] 
# %%
dev_live = pd.merge(dev,gb_live, how='inner',on='종목코드')
dev_live
# %%
dev_live = dev_live.drop(['시장구분','종목명'], axis = 1).drop_duplicates()
dev_live
# %%
# target variable
# sns.countplot(data=dev_fn, x='부도관리구분', palette='bwr')
# plt.show()
print(dev_live.groupby('부도관리구분').size())
print(dev_live.groupby('부도구분').size())
print(dev_live.groupby('관리구분').size())
# %%
del dev, df, df2, df4
# %%
##########################################################
'''
부도기업 재무제표 반입
'''
##########################################################
#%%
import pandas as pd
df_bs20 = pd.read_table('/Users/seonjin/ColabProjects/2020_4Q_BS_20210327040036/2020_사업보고서_01_재무상태표_20210327.txt', encoding='cp949', thousands = ',') 
df_bs19 = pd.read_table('/Users/seonjin/ColabProjects/2019_4Q_BS_20210324040040/2019_사업보고서_01_재무상태표_20210324.txt', encoding='cp949', thousands = ',') 
df_bs18 = pd.read_table('/Users/seonjin/ColabProjects/2018_4Q_BS_20210323040318/2018_사업보고서_01_재무상태표_20210323.txt', encoding='cp949', thousands = ',') 
df_bs17 = pd.read_table('/Users/seonjin/ColabProjects/2017_4Q_BS_20210323040043/2017_사업보고서_01_재무상태표_20210323.txt', encoding='cp949', thousands = ',') 
df_bs16 = pd.read_table('/Users/seonjin/ColabProjects/2016_4Q_BS_20210318040334/2016_사업보고서_01_재무상태표_20210318.txt', encoding='cp949', thousands = ',') 
df_bs15 = pd.read_table('/Users/seonjin/ColabProjects/2015_4Q_BS_20210318040048/2015_사업보고서_01_재무상태표_20210318.txt', encoding='cp949', thousands = ',') 
#%%
df_pl20 = pd.read_table('/Users/seonjin/ColabProjects/2020_4Q_PL_20210410040340/2020_사업보고서_03_포괄손익계산서_연결_20210410.txt', encoding='cp949', thousands = ',') 
df_pl19 = pd.read_table('/Users/seonjin/ColabProjects/2019_4Q_PL_20210410040118/2019_사업보고서_03_포괄손익계산서_연결_20210410.txt', encoding='cp949', thousands = ',') 
df_pl18 = pd.read_table('/Users/seonjin/ColabProjects/2018_4Q_PL_20210330040118/2018_사업보고서_03_포괄손익계산서_연결_20210330.txt', encoding='cp949', thousands = ',') 
df_pl17 = pd.read_table('/Users/seonjin/ColabProjects/2017_4Q_PL_20210323040125/2017_사업보고서_03_포괄손익계산서_연결_20210323.txt', encoding='cp949', thousands = ',') 
df_pl16 = pd.read_table('/Users/seonjin/ColabProjects/2016_4Q_PL_20210318040411/2016_사업보고서_03_포괄손익계산서_연결_20210318.txt', encoding='cp949', thousands = ',') 
df_pl15 = pd.read_table('/Users/seonjin/ColabProjects/2015_4Q_PL_20210318040135/2015_사업보고서_03_포괄손익계산서_연결_20210318.txt', encoding='cp949', thousands = ',') 
#%%
df_cf20 = pd.read_table('/Users/seonjin/ColabProjects/2020_4Q_CF_20210410040410/2020_사업보고서_04_현금흐름표_20210410.txt', encoding='cp949', thousands = ',') 
df_cf19 = pd.read_table('/Users/seonjin/ColabProjects/2019_4Q_CF_20210410040150/2019_사업보고서_04_현금흐름표_20210410.txt', encoding='cp949', thousands = ',') 
df_cf18 = pd.read_table('/Users/seonjin/ColabProjects/2018_4Q_CF_20210330040149/2018_사업보고서_04_현금흐름표_20210330.txt', encoding='cp949', thousands = ',') 
df_cf17 = pd.read_table('/Users/seonjin/ColabProjects/2017_4Q_CF_20210323040159/2017_사업보고서_04_현금흐름표_20210323.txt', encoding='cp949', thousands = ',') 
df_cf16 = pd.read_table('/Users/seonjin/ColabProjects/2016_4Q_CF_20210318040442/2016_사업보고서_04_현금흐름표_20210318.txt', encoding='cp949', thousands = ',') 
df_cf15 = pd.read_table('/Users/seonjin/ColabProjects/2015_4Q_CF_20210318040215/2015_사업보고서_04_현금흐름표_20210318.txt', encoding='cp949', thousands = ',') 
# %%
df = pd.concat([df_bs15,df_bs16,df_bs17,df_bs18,df_bs19,df_bs20,df_pl15,df_pl16,df_pl17,df_pl18,df_pl19,df_pl20,df_cf15,df_cf16,df_cf17,df_cf18,df_cf19,df_cf20], ignore_index=True)
# %%
df.info()
del df_bs15,df_bs16,df_bs17,df_bs18,df_bs19,df_bs20,df_pl15,df_pl16,df_pl17,df_pl18,df_pl19,df_pl20,df_cf15,df_cf16,df_cf17,df_cf18,df_cf19,df_cf20
# %%
# 위에 건전기업 코드 동일. 그대로 돌리기

# %%
##########################################################
'''
관리종목 처리
'''
##########################################################
# %%
gb_mng = pd.read_csv('data_company_관리종목2.csv')
# gb_mng.rename(columns = {'종목명' : '회사명'}, inplace = True)
gb_mng
# %%
gb_mng = gb_mng[['종목코드', '지정일자']]
gb_mng['기준년'] = pd.to_numeric(gb_mng['지정일자'].str[:4]) - 1
gb_mng
# %%
dev['기준년'] = pd.to_numeric(dev['결산기준일'].str[:4])
# %%
dev_mng = pd.merge(dev, gb_mng, how='inner',on=['종목코드','기준년'])
dev_mng
# %%
gb = pd.read_excel('data_company.xlsx', sheet_name = 'Sheet3', engine='openpyxl')
gb_mng2 = gb[gb['부도관리구분'] == 2] 
# %%
dev_mng_fn = pd.merge(dev_mng, gb_mng2, how='inner',on=['종목코드'])
dev_mng_fn
# %%
# dup 삭제 검증
# mm = pd.DataFrame([dev_mng_fn.groupby('종목코드')['회사명'].count()])
# mm.transpose().reset_index()[mm.transpose().reset_index()['회사명'] > 1]
# %%
dev_mng_fn = dev_mng_fn.drop(['시장구분','종목명','지정일자'], axis = 1).drop_duplicates()
dev_mng_fn
# %%
##########################################################
'''
부도기업 부도여부 붙이기
'''
##########################################################
# %%
gb_budo = gb[gb['부도관리구분'] == 1] 
# %%
print(gb_budo.shape)
print(gb_budo.drop_duplicates().shape)
# %%
dev_budo = pd.merge(dev, gb_budo, how='inner',on='종목코드')
dev_budo
# %%
dev_budo.groupby(by=['종목코드','회사명']).count()
# %%
# dev_budo_fn = dev_budo.drop_duplicates(['종목코드','회사명'], keep='last')
dev_budo_fn = dev_budo.drop_duplicates(['종목코드'], keep='last')
# %%
dev_budo_fn = dev_budo_fn.drop(['시장구분','종목명'], axis = 1).drop_duplicates()
dev_budo_fn
# %%
dev_budo_fn
# %%
dev_budo_fn.info()
# %%
dev_budo_fn.isnull().sum() 
# %%
print(dev_budo_fn.shape)
print(dev_budo_fn.drop_duplicates().shape)
# %%
# target variable
# sns.countplot(data=dev_fn, x='부도관리구분', palette='bwr')
# plt.show()
print(dev_budo_fn.groupby('부도관리구분').size())
print(dev_budo_fn.groupby('부도구분').size())
print(dev_budo_fn.groupby('관리구분').size())
# %%
del df, df2, df4, dev, dev_budo
# %%
##########################################################
'''
부도기업 & 건전기업 & 관리종목
'''
##########################################################
# %%
dev_fn = pd.concat([dev_live, dev_budo_fn, dev_mng_fn], ignore_index=True)
dev_fn
# %%
print(dev_fn.groupby('부도관리구분').size())

import matplotlib.pyplot as plt
plt.bar([0,1,2],dev_fn.groupby('부도관리구분').size())
# %%
##########################################################
'''
감성분석 설명변수 도입 (RNN, count)
'''
##########################################################
# %%
df_sent = pd.read_excel('data_company_score_60.xlsx', sheet_name = 'data_company_score_60', engine='openpyxl')
df_sent_cnt = pd.read_excel('word_count_score_60.xlsx',sheet_name = 'word_count_score', engine='openpyxl')
# df_sent = pd.read_csv('data_company_score_6m_mng.csv')
# df_sent['종목코드'] = pd.to_numeric(df_sent['종목코드'])
# %%
df_sent.head()
# %%
df_sent_cnt.head()
# %%
df_sent_mean = df_sent.groupby(['회사','종목코드'])['스코어'].mean()
df_sent_mean = pd.DataFrame(df_sent_mean).drop_duplicates().reset_index()
df_sent_mean
# %%
df_sent_cnt_mean = df_sent_cnt.groupby(['회사','종목코드'])['스코어'].mean()
df_sent_cnt_mean = pd.DataFrame(df_sent_cnt_mean).drop_duplicates().reset_index()
df_sent_cnt_mean.columns = ['회사','종목코드','스코어cnt']
df_sent_cnt_mean
# %%
dev_fn_sent = pd.merge(dev_fn, df_sent_mean, left_on = '종목코드', right_on = '종목코드' , how='inner')
dev_fn_sent
# %%
dev_fn_sent = pd.merge(dev_fn_sent, df_sent_cnt_mean, left_on = '종목코드', right_on = '종목코드' , how='inner')
dev_fn_sent
# %%
print(dev_fn_sent.groupby('부도관리구분').size())
plt.bar([0,1,2],dev_fn_sent.groupby('부도관리구분').size())
# %%
dev_fn = dev_fn_sent
# %%
##########################################################
'''
EDA
'''
##########################################################
# https://www.kaggle.com/rsj0113/simple-yet-powerful-bankrupt-prediction-model/edit/run/58038260
# runtime
import timeit

# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

# preprocessing
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer

# Ml model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

np.warnings.filterwarnings('ignore')

# 한글폰트
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# %%
# 1. Data Loading and Data Cleaning

print(dev_fn.isnull().values.any())
print(dev_fn.shape)
print(dev_fn.groupby('부도관리구분').size())
# %%
# 극단치 치환
dev1 = dev_fn.replace([np.inf, -np.inf], np.nan)
dev1 = dev1.fillna(0)
# %%
# 타겟 정제
dev1['부도관리구분'] = dev1['부도관리구분'].replace(2,1)
dev1 = dev1.drop(['회사명','결산기준일','부도구분','관리구분','기준년','회사_x','회사_y'], axis = 1)
# %%
dev1.head()
# %%
print(dev1.groupby('부도관리구분').size())
# %%
# hist 변수별 분포 
dev1.hist(figsize=(20,20), edgecolor='white')
plt.show()
# %%
dev2 = dev1[dev_fn['자산총계'] < 1000000000000]
# %%
# heatmap
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(dev2.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True)
# %%
# scatterplot 두개의 변수
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

sns.scatterplot(data=dev2, x='이익잉여금/총자산', y='유동부채/총자산', hue='부도관리구분', ax=ax[0])
sns.scatterplot(data=dev2, x='자기자본비율', y='이익잉여금/총자산', hue='부도관리구분', ax=ax[1])
sns.scatterplot(data=dev2, x='스코어', y='스코어cnt', hue='부도관리구분', ax=ax[2])
# %%
# 부도 / 건전기업별 재무비율 평균
central = dev2.groupby('부도관리구분').median().reset_index()
features = list(central.keys()[2:])
display(central)
# central.to_csv('central.csv', encoding='utf-8-sig')
# %%
fig, ax = plt.subplots(7,6, figsize=(20,20))

ax = ax.ravel()
position = 0

for i in features:
    sns.barplot(data=central, x='부도관리구분', y=i, ax=ax[position], palette='bwr')
    position += 1
    
plt.show()

# %%
# Plotting Boxplots of the preprocessed numerical features
# 보완필요
plt.figure(figsize = (20,20))
ax =sns.boxplot(data = dev2[:], orient="h")
ax.set_title('Bank Data Preprocessed Boxplots', fontsize = 18)
ax.set(xscale="log")
plt.show()

# %%
##########################################################
'''
재무비율 모델링
'''
##########################################################
# https://www.kaggle.com/sathianpong/company-bankeuptcy
# %%
# 5. Build the models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
# import xgboost  

import statsmodels.api as sm

# %%
features = [
# '부도관리구분',
# '종목코드',
# '자산총계',
# '재고자산회전율',
'자기자본비율',
'이익잉여금/총자산',
# '유동부채/총자산',
'현금/총자산',
'유동자산/총자산',
# '현금흐름/총자산', 
# '총자산회전율',
'총자산순이익율',
# '총부채/총자산',
# '순운전자본/총자본',
# '당기순이익/총자산',
'총자본순이익율',
'고정비율',
# '고정장기적합율',
# '부채비율',
# '유동부채비율',
# '고정부채비율',
# '현금흐름/총자본',
# '총자본회전율',
'현금비율',
# '유동비율',
'이익잉여금/총부채',
# '고정자산회전율',
# '영업활동이익/총부채',
# '금융비용/총부채',
'유동부채/총부채',
# '현금흐름/부채',
# '매출액순이익율',
# '매출액영업이익율',
# '매출액이자비용율',
# '현금흐름/매출액',
# '유동부채/매출액',
# '이익잉여금/유동자산',
'총자산증가율',
'유형고정자산증가율',
# '매출액증가율',
# '순이익증가율',
# '고정자산증가율',
# '스코어',
# '스코어cnt',
'부도관리구분'
]
X = dev2[features].values
y = dev2.iloc[:,-3].values.reshape(-1, 1)
# %%
# 부도 / 건전기업별 재무비율 평균
central = dev2[features].groupby('부도관리구분').median().reset_index()
features = list(central.keys()[2:])
display(central)
central.to_csv('central.csv', encoding='utf-8-sig')
# %%
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(dev2[features].corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), annot=True)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, stratify=y)
scaler = StandardScaler().fit(X_train)

# %%
models = dict()
# 현재 관리구분이 2 여서 1>2로 바꿔줌
models['Logreg'] = LogisticRegression(penalty='elasticnet',  class_weight={0:1,1:3}, solver='saga', l1_ratio=0.7)
models['SVM'] = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
models['Random Forest'] = RandomForestClassifier(n_estimators=300, class_weight={0:1,1:3})
models['GradientBoost'] = GradientBoostingClassifier(n_estimators=300)
# models['AdaBoost'] = AdaBoostClassifier(n_estimators=300)
# models['XGBoost'] = xgboost.XGBClassifier()
# models['KNN'] = KNeighborsClassifier(n_neighbors=i)
# models['GBRT'] = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=1)
# models['Logit']  = sm.Logit
# %%
for model in models:
    if model == 'Logreg':
        train = scaler.transform(X_train)
    else:
        train = X_train
    models[model].fit(train, y_train)
    print(model + ' : fit')
# %%
# %%
# 5.2 Performance in test set
for x in models:   
    if x == 'Logreg':
        test = scaler.transform(X_test)
    else:
        test=X_test
    print('------------------------'+x+'------------------------')
    model = models[x]
    y_test_pred = model.predict(test)
    arg_test = {'y_true':y_test, 'y_pred':y_test_pred}
    print(confusion_matrix(**arg_test))
    print(classification_report(**arg_test))     
# %%
# 6. ROC Curve
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots()
fig.set_size_inches(13,6)

for m in models:
    y_pred = models[m].predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred[:,1].ravel())
    roc_auc = auc(fpr, tpr)
    print("%s Area under the ROC curve : %f" % (m, roc_auc))
    plt.plot(fpr,tpr, label=m)

plt.xlabel('False-Positive rate')
plt.ylabel('True-Positive rate')
plt.legend()
plt.show()
# %%
df_proba = pd.DataFrame(model.predict_proba(test))
df_proba 
# %%
df_proba['pred'] = model.predict(test)
# %%
# 배열형태로 반환
ft_importance_values = model.feature_importances_

# 정렬과 시각화를 쉽게 하기 위해 series 전환
ft_series = pd.Series(ft_importance_values, index = X_train)
ft_top20 = ft_series.sort_values(ascending=False)[:20]

# 시각화
plt.figure(figsize=(8,6))
plt.title('Feature Importance Top 20')
sns.barplot(x=ft_top20, y=ft_top20.index)
plt.show()