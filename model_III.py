import numpy as np
import pandas as pd                                           
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing#, cross_validation
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier 
from scipy import stats
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from warnings import filterwarnings
filterwarnings('ignore')
sklearn.__version__
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Functions

# Compute p-value using the chi-squared test for binary predictors
def chi2_cols(y,x):
    '''
    input:
    y: 1-d binary label array
    x: 1-d binary feature array
    
    return:
    chi2 statistic and p-value
    '''
    y_list=y.astype(int).tolist()
    x_list=x.astype(int).tolist()
    freq=np.zeros([2,2])
    for i in range(len(y_list)):
        if y_list[i]==0 and x_list[i]==0:
            freq[0,0]+=1
        if y_list[i]==1 and x_list[i]==0:
            freq[1,0]+=1
        if y_list[i]==0 and x_list[i]==1:
            freq[0,1]+=1
        if y_list[i]==1 and x_list[i]==1:
            freq[1,1]+=1
    y_0_sum=np.sum(freq[0,:])
    y_1_sum=np.sum(freq[1,:])
    x_0_sum=np.sum(freq[:,0])
    x_1_sum=np.sum(freq[:,1])
    total=y_0_sum+y_1_sum
    y_0_ratio=y_0_sum/total
    freq_=np.zeros([2,2])    
    freq_[0,0]=x_0_sum*y_0_ratio
    freq_[0,1]=x_1_sum*y_0_ratio
    freq_[1,0]=x_0_sum-freq_[0,0]
    freq_[1,1]=x_1_sum-freq_[0,1]
    stat,p_value=stats.chisquare(freq,freq_,axis=None)    
    return p_value#, stat

# Compute the variables statistics such as mean, std, p-value, and correlation 
def stat_test(df, y):
    '''
    input:
    df: 2-d dataframe of the dataset
    y: 1-d binary label array 
    
    return:
    a dataframe of the variables statistics such as mean, std, p-value, and correlation 
    '''
    name = pd.DataFrame(df.columns,columns=['Variable'])
    df0=df[y==0]
    df1=df[y==1]
    pvalue=[]
    y_corr=[]
    for col in df.columns:
        if df[col].nunique()==2:
            # Compute p-value using the chi-squared test for binary predictors
            pvalue.append(chi2_cols( y,df[col]))
        else:
            # Compute p-value using the Kolmogorov-Smirnov test for continuous predictors
            pvalue.append(stats.ks_2samp(df0[col], df1[col]).pvalue)
        
        # Compute pairwise correlation of the variable and the label
        y_corr.append(df[col].corr(y))
    name['All_mean']=df.mean().values
    name['y1_mean']=df1.mean().values
    name['y0_mean']=df0.mean().values
    name['All_std']=df.std().values
    name['y1_std']=df1.std().values
    name['y0_std']=df0.std().values
    name['p-value']=pvalue
    name['y_corr']=y_corr
    return name.sort_values(by=['p-value'])

# Compute pairwise correlation of each continuous variable and the label
# and drop the variable with low correlation (<0.04)
def df_ycorr(df,col_y):
    '''
    input:
    df: 2-d dataframe of the dataset
    col_y: label name
    
    return:
    a list of variable names that we want to drop because of low correlation with outcome
    '''
    drop_cols=[]
    for col in df.columns:
        if df[col].nunique()!=2:
            y_corr=round(df[col_y].corr(df[col]),2)
            if (abs(y_corr)<0.04):
                drop_cols.append(col)
    return drop_cols

# Compute pairwise correlation of variables with each other
# and if the correlation is high (>0.8), we keep one variable of the highly-correlated variables
def high_corr(df, thres=0.8):
    '''
    input:
    df: 2-d dataframe of the dataset
    thres: Threshold we consider to determine highly correlated variables 
    
    return:
    a list of pairs of two highly correlated variables
    '''
    corr_matrix_raw = df.corr()
    corr_matrix = corr_matrix_raw.abs()
    high_corr_var_=np.where(corr_matrix>thres)
    high_corr_var=[(corr_matrix.index[x],corr_matrix.columns[y], corr_matrix_raw.iloc[x,y]) for x,y in zip(*high_corr_var_) if x!=y and x<y]
    return high_corr_var

# a function to drop variables we want to drop 
def df_drop(df_new, drop_cols):
    '''
    input:
    df_new: 2-d dataframe of the dataset after preprocessing icluding one-hot encoding and statistical feature selection
    drop_cols: a list variables we want to drop 
    
    return:
    our dataframe after dropping variables we want to drop 
    '''
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)


# step1: load dataset (dataset after pre-processing and one-hot encoding)
df_new=pd.read_csv('df_PRESTO_id.csv')

## apply limitations we want

### keep only participants with no more than one menstrual cycle of pregnancy attempt at study entry
df_new=df_new.loc[df_new['ttp_entry']<2,:].reset_index(drop=True)#.head()

### define the time variable for the survival model
df_new['time_variable'] = (df_new['TTP_exit']-df_new['ttp_entry']).astype(int)

### Model III pedict the probability of pregnancy within each menstrual cycle for up to 12 cycles of follow-up (fecundability model)
which_month=12

### some considerations
df_new['b_livebirths']=round(df_new['b_livebirths']) 
for col in ['b_birthorder','b_livebirths']:
    df_new.loc[df_new[col]>3,col]=3  
for col in ['b_csectiontotal']:
    df_new.loc[df_new[col]>2,col]=2    

### drop columns we don't need    
drop_cols=['b_finisheddate_year','persons','TTP_exit','newid']
df_new=df_drop(df_new, drop_cols)        
    
## load variable descriptions to be added to the final predictors' coefficient tables    
explain_Variables=pd.read_csv('variable_list_20200910.csv')


# step2: Statistical Feature Selection

## drop variables with low std (<0.0001)
col_y='pregnant'
df_std = df_new.std()
drop_cols = df_std[df_std<0.0001].index.values
df_new=df_drop(df_new, drop_cols)

## keep one var among high correlated vars with threshold=thres          
high_corr_features = high_corr(df_new, thres=0.8)
drop_cols=['b_everpregnant', 'hxinfert', 'b_geog5_6.0','b_livebirths']#select one to drop among each pair in high_corr_list #for our model_II: drop_cols=['b_everpregnant', 'b_livebirths', 'hxinfert']
df_new=df_drop(df_new, drop_cols) 


# step3: lifelines CoxPHFitter

from lifelines import CoxPHFitter

#scaling/normalizing
from sklearn import preprocessing
col_y = 'pregnant'
scaler = preprocessing.StandardScaler()#MinMaxScaler    
X = df_new.drop([col_y, 'time_variable', 'ttp_entry'], axis=1)
name_cols=df_new.drop([col_y, 'time_variable', 'ttp_entry'], axis=1).columns.values 
X = scaler.fit_transform(X)
df = pd.DataFrame(X, columns=name_cols)
df[col_y] = df_new[col_y]
df['time_variable'] = df_new['time_variable']
df['ttp_entry'] = df_new['ttp_entry']

## drop variables with high p-value (>0.05)
cph = CoxPHFitter()#penalizer=0.01, l1_ratio=1.0)
cph.fit(df=df, duration_col='time_variable', event_col='pregnant',strata='ttp_entry')#OR: duration_col='TTP_exit',entry_col='ttp_entry'
df_coef_cox = cph.summary
df_coef_cox['Variable'] = df_coef_cox.index
df_coef_cox['coef_abs'] = df_coef_cox['coef'].abs()
LRresult_ = pd.merge(df_coef_cox.sort_values(by=['coef_abs'],ascending=False), explain_Variables[['Variable','Label']], how='left', on=['Variable'])
LRresult_.drop(['coef_abs'], axis=1)
vars_small_pvalue = LRresult_[(LRresult_['p']<0.05)]['Variable'].tolist()

my_df = df[vars_small_pvalue + ['time_variable','pregnant','ttp_entry']].copy()

# compute the concordance index 
#concordance_index = []
my_seeds=range(2016, 2021) # the random_state that controls the shuffling applied to the data before applying the split
for seed in my_seeds:

    # split dataset to train-set and test-set
    cph_train, cph_test = train_test_split(my_df, test_size=0.2, random_state=seed) 
    
    # training model on train-set
    cph.fit(df=cph_train, duration_col='time_variable', event_col='pregnant',strata='ttp_entry')

    # Evaluating cox model on test-set
    #concordance_index.append(cph.score(cph_test, scoring_method='concordance_index'))
    concordance_index = cph.score(cph_test, scoring_method='concordance_index')

#print("concordance_index is:" + str(sum(concordance_index) / len(concordance_index)))
print("concordance_index on test is:" + str(concordance_index))

# Feature Selection using train set
X = cph_train.copy()
y=X[["pregnant","time_variable", "ttp_entry"]]
X.drop(["pregnant","time_variable", "ttp_entry"], axis=1, inplace=True)
n_features = X.shape[1]
scores = np.empty(n_features)
m = CoxPHFitter()
for j in range(n_features):
    Xj = X.iloc[:, j:j+1]
    Xj=pd.merge(Xj, y,  how='right', left_index=True, right_index=True)
    m.fit(Xj, duration_col="time_variable", event_col="pregnant", strata='ttp_entry')#, show_progress=True)
    scores[j] = m.score(Xj, scoring_method='concordance_index')
my_df_predictors = pd.Series(scores, index=X.columns).sort_values(ascending=False)
k=15
top15_predictors = my_df_predictors[0:k].index.tolist()

# Repeat with top15 predictors
my_top15_train = cph_train[top15_predictors + ['time_variable','pregnant','ttp_entry']].copy()
cph = CoxPHFitter()#penalizer=0.01, l1_ratio=1.0)
cph.fit(df=my_top15_train, duration_col='time_variable', event_col='pregnant',strata='ttp_entry')#OR: duration_col='TTP_exit',entry_col='ttp_entry'
df_coef_cox = cph.summary
df_coef_cox['Variable'] = df_coef_cox.index
df_coef_cox['coef_abs'] = df_coef_cox['coef'].abs()
LRresult_ = pd.merge(df_coef_cox.sort_values(by=['coef_abs'],ascending=False), explain_Variables[['Variable','Label']], how='left', on=['Variable'])
LRresult_ = LRresult_.drop(['coef_abs'], axis=1)

# Evaluating top15 cox model on test
top15_columns = my_top15_train.columns.tolist()
test_concordance_index = cph.score(cph_test[top15_columns], scoring_method='concordance_index')
print(test_concordance_index)

# model_III top variables
LRresult_
