#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 2021

@author: Zahra Zad <zad@bu.edu>
@author: Taiyao Wang
"""
"""
To use the code, you can go through the steps explained bellow and follow the detailed instructions commented in the script:

Step 1: Start with the section “load dataset”:
-Load the dataset including the features and the label, in the format of pandas dataframe

Step 2: Go through the section “Statistical Feature Selection”

Step 3: Go through the section “implementation of the models” to implement:
-Full models (i.e., least parsimonious) that contain all variables selected after sStatistical Feature Selection
-Sparse models that contain variables selected after both Statistical Feature Selection and Recursive Feature Elimination
-Parsimonious models that limit recursive feature elimination to select a model with up to 15 variables
"""

#import modules we need

import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, precision_recall_curve, plot_precision_recall_curve 
from sklearn.ensemble import RandomForestClassifier                                             
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier 
import lightgbm as lgb
from sklearn.feature_selection import RFE,RFECV
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import math
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

# train a model only using the training set:
# tune the model hyperparameters via cross-validation and returns the model with the best cross-validation score fitted on the whole training set
def my_train(X_train, y_train, model='LR', penalty='l1', cv=5, scoring='roc_auc', class_weight= 'balanced',seed=2020):    
    '''
    input:
    X_train: 2-d array of the training set except the label
    y_train: 1-d array training set label
    model: Type of algorithm we want to develop:  'LR', 'SVM', 'MLP', 'LR', or 'LGB'
    penalty: Regularization norm for linear models LR and SVM:  'l1' or 'l2'
    cv: Number of folds in cross-validation
    scoring: Strategy to evaluate the performance of the cross-validated model on the validation set: 'roc_auc', 'f1', etc
    class_weight: Weights associated with classes
    seed: random_state used to shuffle the data
    
    return:
    the model with the best cross-validation score fitted on the whole training dataset
    '''
    # use the training dataset to tune the model hyperparameters via cross-validation 
    # Support Vector Machine algorithm
    if model=='SVM':
        svc=LinearSVC(penalty=penalty, class_weight= class_weight, dual=False, max_iter=5000)#, tol=0.0001
        param_grid = {'C':[0.001,0.01,0.1,1,10]} #'kernel':('linear', 'rbf'),
        gsearch = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring)
    # Boosted Trees algorithm
    elif model=='LGB':        
        param_grid = {
            'feature_fraction': 0.4,
            'bagging_fraction': [0.9],   
            'nthread': [3],
            'num_leaves': range(6,12,2),
            'min_data_in_leaf': range(14,26,2),
            'learning_rate': [0.08,0.10,0.12,0.14], #0.01*range(8,15,2),
            'feature_fraction': [0.2,0.3,0.4,0.5,0.6] #0.1*range(2,7,1)
        }
        lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', class_weight= class_weight, random_state=seed)# eval_metric='auc' num_boost_round=2000,learning_rate=0.1,
        gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=scoring)  
    # neural network algorithm
    elif model=='MLP':
        mlp=MLPClassifier(random_state=seed, tol=0.01)
        param_grid = {'hidden_layer_sizes':[(C, ),(round(C/2), 2),(round(C/4), 4)] for C in [32,64,128,256,512]}
        gsearch = GridSearchCV(mlp, param_grid, cv=cv, scoring=scoring)
    # Logistic Regression algorithm
    else:
        LR = LogisticRegression(penalty=penalty, class_weight= class_weight,solver='liblinear', random_state=seed)
        parameters = {'C':[0.001,0.01,0.1,1,10] }
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring)
    
    # fit the model with the best cross-validation score on the whole training dataset
    gsearch.fit(X_train, y_train)
    clf=gsearch.best_estimator_
    #print('Best parameters found by grid search are:', gsearch.best_params_)
    
    # returns the model with the best cross-validation score fitted on the whole training dataset
    return clf

# find optimal threshold that leads to the highest 'weighted_F1_score' among thresholds on the decision function used to compute fpr and tpr of the training set
def cal_f1_scores(y, y_pred_score):
    '''
    input: 
    y: Ground truth target values of the training set
    y_pred_score: Target scores of the training set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    
    return:
    Optimal threshold that leads to the highest 'weighted_F1_score' among thresholds on the decision function used to compute fpr and tpr of the training set
    '''
    # compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    thresholds = sorted(set(thresholds))
    metrics_all = []
    
    for thresh in thresholds:
        y_pred = np.array((y_pred_score > thresh))
        metrics_all.append(( thresh,auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted')))
    
    metrics_df = pd.DataFrame(metrics_all, columns=['thresh','AUC',  'micro_F1_score', 'macro_F1_score','weighted_F1_score'])
    
    # returns the optimal threshold that leads to the highest 'weighted_F1_score' among thresholds on the decision function used to compute fpr and tpr of the training set
    return metrics_df.sort_values(by = 'weighted_F1_score', ascending = False).head(1)['thresh'].values[0]

# compute performance metrics evaluated on the test set
def cal_f1_scores_te(y, y_pred_score,thresh):
    '''
    input:
    y: Ground truth target values of the test set
    y_pred_score: Target scores of the test set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    
    return:
    a dataframe of performance metrics evaluated on the test set
    '''
    # compute Receiver operating characteristic (ROC)
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    
    # compute the estimated targets 
    # the estimated target is 1 if the target score is greater than the optimal threshold on the decision function found using the training set
    y_pred = np.array((y_pred_score > thresh))
    
    if if_RFE==0: # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE
        metrics_all = [ (thresh, auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted'), precision_score(y, y_pred, average='weighted'), recall_score(y, y_pred, average='weighted'))]
        metrics_df = pd.DataFrame(metrics_all, columns=['thresh','AUC','micro_F1_score','macro_F1_score','weighted_F1_score','weighted_precision_score','weighted_recall_score'])
    else:
        metrics_all = [ (thresh,auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'),f1_score(y, y_pred, average='weighted'))]
        metrics_df = pd.DataFrame(metrics_all, columns=['thresh','AUC',  'micro_F1_score', 'macro_F1_score','weighted_F1_score'])
    
    # returns a dataframe of performance metrics evaluated on the test set
    return metrics_df

# test the obtained model on the test set
def my_test(X_train, xtest, y_train, ytest, clf, target_names, model='LR'):
    '''
    input:
    X_train: 2-d array of the training set except the label
    xtest: 2-d array of the test set except the label
    y_train: 1-d array training set label
    ytest: 1-d array test set label
    clf: the model with the best cross-validation score fitted on the whole training dataset
    target_names: 0 and 1 as the label is binary: ['0', '1'] 
    model: Type of algorithm we want to develop:  'LR', 'SVM', 'MLP', 'LR', or 'LGB'
    
    return:
    a dataframe of performance metrics evaluated on the test set
    '''
    # compute target scores of the training set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    if model=='SVM':
        ytrain_pred_score=clf.decision_function(X_train) 
    else:
        ytrain_pred_score=clf.predict_proba(X_train)[:,1] 
    
    # find the optimal threshold on the decision function used to compute fpr and tpr
    thres_opt=cal_f1_scores( y_train, ytrain_pred_score)
    
    # compute target scores of the test set that can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    if model=='SVM':
        ytest_pred_score=clf.decision_function(xtest)
    else:
        ytest_pred_score=clf.predict_proba(xtest)[:,1] 
    
    # returns a dataframe of performance metrics evaluated on the test set
    return cal_f1_scores_te(ytest, ytest_pred_score,thres_opt)


# develop different ML algorithms 'LR', 'SVM', 'MLP', 'LR', or 'LGB'
def tr_predict(df_new, col_y, target_names = ['0', '1'], model='LR',penalty='l1', cv_folds=5,scoring='roc_auc', test_size=0.2):
    '''
    input:
    df_new: 2-d dataframe of the dataset after preprocessing icluding one-hot encoding and statistical feature selection
    col_y: Labe name 
    target_names: 0 and 1 as the label is binary: ['0', '1'] 
    model: Type of algorithm we want to develop:  'LR', 'SVM', 'MLP', 'LR', or 'LGB'
    penalty: Regularization norm for linear models LR and SVM:  'l1' or 'l2' 
    cv_folds: Number of folds in cross-validation
    scoring: Strategy to evaluate the performance of the cross-validated model on the validation set: 'roc_auc', 'f1', etc
    test_size: Proportion of the dataset to include in the test split
    
    return:
    a dataframe including the predictors' coefficients and statistics based on the selected algorithm
    '''
    # Standardize features by removing the mean and scaling to unit variance
    scaler = preprocessing.StandardScaler()#MinMaxScaler    
    
    y= df_new[col_y].values # 1-d binary label array 
    metrics_all=[] # a list to keep metrics calculated on the test set for each run
    
    my_seeds=range(2020, 2025) # the random_state that controls the shuffling applied to the data before applying the split
    for seed in my_seeds: # we repeat the model development 5 times and we use a different seed for each run
        
        X = df_new.drop([col_y], axis=1).values # dataset excluding the label in the format of 2-d array
        name_cols=df_new.drop([col_y], axis=1).columns.values # features names
        
        # Fits transformer to X and returns a transformed version of X
        X = scaler.fit_transform(X)
        
        # Split the dataset to five random parts, where four parts constituted the training dataset, and the fifth part constituted the testing dataset
        X_train, xtest, y_train, ytest = train_test_split(X, y, stratify=y, test_size=test_size,  random_state=seed)# Split arrays into random train and test subsets
        
        # train a model only using the training set
        clf = my_train(X_train, y_train, model=model, penalty=penalty, cv=cv_folds, scoring=scoring, class_weight= 'balanced',seed=seed)    
        
        # test the obtained model on the test set
        metrics_all.append(my_test(X_train, xtest, y_train, ytest, clf, target_names, model=model))
    
    # compute the mean and standard deviation of the model performance statistics across these five runs
    metrics_df=pd.concat(metrics_all)
    if if_RFE==0: # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE
        metrics_df = metrics_df[['AUC','micro_F1_score','weighted_F1_score','weighted_precision_score','weighted_recall_score']].describe().T[['mean','std']].stack().to_frame().T
    else:
        metrics_df = metrics_df[['AUC','micro_F1_score','weighted_F1_score']].describe().T[['mean','std']].stack().to_frame().T       
    
    #create the dataframe of the predictors' coefficients based on model type
    if model=='LGB': 
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.feature_importances_,2))),columns=['Variable','coef_'])        
        fig, ax = plt.subplots()
        lgb.plot_importance(clf, ax=ax, max_num_features=100)
        plt.title("Light GBM Feature Importance")
    elif model=='MLP':
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.coefs_[0][:,0],2))),columns=['Variable','coef_'])
    else:
        df_coef_=pd.DataFrame(list(zip(name_cols, np.round(clf.coef_[0],5))),columns=['Variable','coef_']) 
        if ((model=='LR') & (if_RFE==0)): 
            #plot_precision_recall_curve
            disp = plot_precision_recall_curve(clf, xtest, ytest)
            disp.ax_.axis(ymin=0,ymax=1)
            disp.ax_.set_title('2-class Precision-Recall curve')            
            #calculate standard_errors 
            predProbs = clf.predict_proba(X_train)
            X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            V = np.product(predProbs, axis=1)
            covLogit = np.linalg.pinv(np.dot((X_design.T * V), X_design))
            standard_errors = np.sqrt(np.diag(covLogit))
            df_coef_['standard_errors'] = standard_errors[1:]                
    df_coef_['coef_abs']=df_coef_['coef_'].abs()    
    
    # return two dataframes
    # one dataframe including the predictors' coefficients and statistics based on the selected algorithm
    # the other dataframe including metrics (mean and std) evaluated on the test set
    if ((model=='LR') & (if_RFE==0)):#when we want to report predicotrs' satandard_errors as well as their coefficients
        return df_coef_.sort_values('coef_abs', ascending=False)[['coef_','Variable','standard_errors']], metrics_df#, scaler
    else:
        return df_coef_.sort_values('coef_abs', ascending=False)[['coef_','Variable']], metrics_df#, scaler

# Featture selection by Recursive Feature Elimination (select features by recursively considering smaller and smaller sets of features)
def my_RFE(df_new, col_y, my_range = range(1,11), my_C_range=[0.01,0.1,1], my_penalty='l1', class_weight='balanced', solver='liblinear'):            
    '''
    input:
    df_new: 2-d dataframe of the dataset after preprocessing icluding one-hot encoding and statistical feature selection
    col_y: Label's name 
    my_range: Range of the desired numbers of features we want to be selected finally
    my_C_range: Range of hyperparameter C of the LR model which we use as the estimator in RFE
    my_penalty: Norm of the penalty for the LR models which we use as the estimator in RFE
    class_weight: Weights associated with classes
    solver: Algorithm to use in the optimization problem
    
    return:
    our dataframe after featture selection by recursive feature elimination
    '''
    metric_all_rfe = []
    Xraw = df_new.drop(col_y, axis=1).values
    y = df_new[col_y].astype(int)
    
    for my_C in my_C_range: # try different hyperparameter C with the LR model which we use as the estimator in RFE
        
        for n_select in my_range: # try different numbers of features to find how many features result in best performance
            
            # Standardize features by removing the mean and scaling to unit variance
            scaler = preprocessing.StandardScaler()#MinMaxScaler
            
            # Fits transformer to X and returns a transformed version of X
            X = scaler.fit_transform(Xraw)
            
            # the LR model which we use as the estimator in RFE
            clf = LogisticRegression(C=my_C, penalty=my_penalty, tol=0.01, class_weight=class_weight, solver=solver)
            
            # select features by recursively considering smaller and smaller sets of features
            rfe = RFE(clf, n_select, step=1)
            rfe.fit(X, y.ravel())
            
            # Selected (i.e., estimated best) features are assigned rank 1
            # so we drop features ranked greater than 1  
            X=df_new.drop(names[rfe.ranking_>1], axis=1)
            
            # evaluate the dataset of selected features using 'LR' model with 'l2' norm regularization
            df_coef_RFE, metric_df_RFE=tr_predict(X, col_y, target_names = ['0', '1'], model='LR', penalty='l2', cv_folds=5,scoring='roc_auc', test_size=0.2)
            metric_all_rfe.append([my_C, n_select]+metric_df_RFE.values.tolist()[0])
    
    metric_all_rfe = pd.DataFrame(metric_all_rfe, columns=['my_C','n_select','AUC-mean','AUC-std','micro_F1_score-mean','micro_F1_score-std','weighted_F1_score-mean','weighted_F1_score-std'])
    
    # we pick the my_C and n_select that lead to the model with highest 'AUC-mean' minus 'AUC-std' 
    metric_all_rfe['AUC_'] = metric_all_rfe['AUC-mean'] - metric_all_rfe['AUC-std']
    scaler = preprocessing.StandardScaler()#MinMaxScaler
    X = scaler.fit_transform(Xraw)
    clf = LogisticRegression(C=metric_all_rfe.loc[metric_all_rfe['AUC_'].idxmax(),'my_C'],penalty=my_penalty,tol=0.01,class_weight='balanced',solver='liblinear')
    rfe = RFE(clf, metric_all_rfe.loc[metric_all_rfe['AUC_'].idxmax(),'n_select'], step=1)
    rfe.fit(X, y.ravel())
    X = df_new.drop(names[rfe.ranking_>1], axis=1)
    return X  # our dataframe after featture selection by recursive feature elimination



# step1: load dataset (dataset after pre-processing and one-hot encoding)

df_new=pd.read_csv('df_PRESTO_id.csv')

## apply limitations we want

### keep only participants with no more than one menstrual cycle of pregnancy attempt at study entry
df_new=df_new.loc[df_new['ttp_entry']<2,:].reset_index(drop=True)

### Model I pedict the probability of pregnancy in fewer than 12 menstrual cycles of pregnancy attempt time (infertility model)
### Model II pict the probability of pregnancy in fewer than 7 menstrual cycles of pregnancy attempt time (subfertility model)
which_month=12 #for model_II: which_month=7

### set y(ttp_exit==12)=0 
df_new.loc[(df_new['TTP_exit']>=which_month),['pregnant']]=0
### exclude all women who dropped before 11 months without becoming pregnant
df_new = df_new[ (df_new['TTP_exit']>=which_month) | ((df_new['TTP_exit']<which_month) & (df_new['pregnant']==1))]
df_new = df_new.reset_index(drop=True)

### some considerations
df_new['b_livebirths']=round(df_new['b_livebirths']) 
for col in ['b_birthorder','b_livebirths']:
    df_new.loc[df_new[col]>3,col]=3
for col in ['b_csectiontotal']:
    df_new.loc[df_new[col]>2,col]=2 
    
### drop columns we don't need
drop_cols=['newid','TTP_exit','b_finisheddate_year']
df_new=df_drop(df_new, drop_cols)

## load variable descriptions to be added to the final predictors' coefficient tables
explain_Variables=pd.read_csv('variable_list_20200910.csv')


# step2: Statistical Feature Selection

## drop variables with low std (<0.0001)
col_y='pregnant'
df_std = df_new.std()
drop_cols = df_std[df_std<0.0001].index.values
df_new=df_drop(df_new, drop_cols)

## drop variables with high p-value (>0.05)          
result=stat_test(df_new, df_new[col_y])
drop_cols=result.loc[result['p-value']>0.05,'Variable'].values
df_new=df_drop(df_new, drop_cols)

## drop continuous variables with low correlation with y 
drop_cols=df_ycorr(df_new,col_y)
df_new=df_drop(df_new, drop_cols)

## keep one var among high correlated vars with threshold=thres          
high_corr_features = high_corr(df_new, thres=0.8)
print(high_corr_features)
drop_cols=['b_livebirths', 'hxinfert']#select one to drop among each pair in high_corr_list #for our model_II: drop_cols=['b_everpregnant', 'b_livebirths', 'hxinfert']
df_new=df_drop(df_new, drop_cols) 


# step3: implementation of the models

## full models: (i.e., least parsimonious) contain all variables selected after statistical feature selection

y = df_new[col_y].astype(int)
names = df_new.drop(col_y, axis=1).columns 
if_RFE=0 # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE

### Logistic Regression with l2 norm regularization (L2LR)
df_coef_L2, metrics_df_L2=tr_predict(df_new, col_y, target_names=['0', '1'], model='LR',penalty='l2',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df_bf=metrics_df_L2

### Logistic Regression with l1 norm regularization (L1LR)
df_coef_L1, metrics_df_L1=tr_predict(df_new, col_y, target_names=['0', '1'], model='LR', penalty='l1',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df_bf=pd.concat([metrics_df_bf, metrics_df_L1])

### Support Vector Machines with l1 norm regularization (L1SVM)
df_coef_svmL1, metrics_df_svmL1=tr_predict(df_new, col_y, target_names=['0', '1'], model='SVM', penalty='l1',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df_bf=pd.concat([metrics_df_bf, metrics_df_svmL1])

### Support Vector Machines with l2 norm regularization (L2SVM)
df_coef_svmL2, metrics_df_svmL2=tr_predict(df_new, col_y, target_names=['0', '1'], model='SVM',penalty='l2',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df_bf=pd.concat([metrics_df_bf, metrics_df_svmL2])

### feed forward Multilayer Perceptron Neural Networks (MLP)
df_coef_mlp, metrics_df_mlp=tr_predict(df_new, col_y, target_names=['0', '1'], model='MLP', cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df_bf=pd.concat([metrics_df_bf, metrics_df_mlp])

### Gradient Boosted Decision Trees, Light Gradient Boosting Machine (LightGBM)
df_coef_LGB, metrics_df_LGB=tr_predict(df_new, col_y, target_names=['0', '1'], model='LGB', cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df_bf=pd.concat([metrics_df_bf, metrics_df_LGB])

# print performance metrics of the full models
metrics_full= metrics_df_bf.set_index(pd.Index(['L2LR','L1LR','L1SVM','L2SVM','NN','LGB']))
print(metrics_full)


## sparse models: contain variables selected after both statistical feature selection and Recursive Feature Elimination (RFE)

if_RFE=1 # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE

# Featture selection by Recursive Feature Elimination 
my_range = range(1,len(df_new.columns))
X = my_RFE(df_new, col_y, my_range=my_range, my_C_range=[0.01,0.1,1], my_penalty='l1', class_weight='balanced', solver='liblinear')            

if_RFE=0 # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE

### Logistic Regression with l2 norm regularization (L2LR)
df_coef_L2, metrics_df_L2=tr_predict(X, col_y, target_names=['0', '1'], model='LR',penalty='l2',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df= metrics_df_L2

### Logistic Regression with l1 norm regularization (L1LR)
df_coef_L1, metrics_df_L1=tr_predict(X, col_y, target_names=['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_L1])

### Support Vector Machines with l1 norm regularization (L1SVM)
df_coef_svmL1, metrics_df_svmL1=tr_predict(X, col_y, target_names=['0', '1'], model='SVM',penalty='l1',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_svmL1])

### Support Vector Machines with l2 norm regularization (L2SVM)
df_coef_svmL2, metrics_df_svmL2=tr_predict(X, col_y, target_names=['0', '1'], model='SVM',penalty='l2',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_svmL2])

### feed forward Multilayer Perceptron Neural Networks (MLP)
df_coef_mlp, metrics_df_mlp=tr_predict(X, col_y, target_names=['0', '1'], model='MLP',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_mlp])

### Gradient Boosted Decision Trees, Light Gradient Boosting Machine (LightGBM)
df_coef_LGB, metrics_df_LGB=tr_predict(X, col_y, target_names=['0', '1'], model='LGB',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_LGB])

# print performance metrics of the sparse models
metrics_sparse= metrics_df.set_index(pd.Index(['L2LR','L1LR','L1SVM','L2SVM','NN','LGB']))
print(metrics_sparse)


## parsimonious models: limit recursive feature elimination to select a model with up to 15 variables

if_RFE=1 # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE

# Limit RFE Featture selection to select at most 15 features 
my_range = range(1,16)
X = my_RFE(df_new, col_y, my_range=my_range, my_C_range=[0.01,0.1,1], my_penalty='l1', class_weight='balanced', solver='liblinear')            

if_RFE=0 # variable 'if_RFE' is defined only for handling which metrics reported if we are doing RFE or if we are not doing RFE

### Logistic Regression with l2 norm regularization (L2LR)
df_coef_L2, metrics_df_L2=tr_predict(X, col_y, target_names=['0', '1'], model='LR',penalty='l2',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df= metrics_df_L2

### Logistic Regression with l1 norm regularization (L1LR)
df_coef_L1, metrics_df_L1=tr_predict(X, col_y, target_names=['0', '1'], model='LR',penalty='l1',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_L1])

### Support Vector Machines with l1 norm regularization (L1SVM)
df_coef_svmL1, metrics_df_svmL1=tr_predict(X, col_y, target_names=['0', '1'], model='SVM',penalty='l1',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_svmL1])

### Support Vector Machines with l2 norm regularization (L2SVM)
df_coef_svmL2, metrics_df_svmL2=tr_predict(X, col_y, target_names=['0', '1'], model='SVM',penalty='l2',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_svmL2])

### feed forward Multilayer Perceptron Neural Networks (MLP)
df_coef_mlp, metrics_df_mlp=tr_predict(X, col_y, target_names=['0', '1'], model='MLP',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_mlp])

### Gradient Boosted Decision Trees, Light Gradient Boosting Machine (LightGBM)
df_coef_LGB, metrics_df_LGB=tr_predict(X, col_y, target_names=['0', '1'], model='LGB',cv_folds=5,scoring='roc_auc', test_size=0.2)
metrics_df=pd.concat([metrics_df, metrics_df_LGB])

# print performance metrics of the parsimonious models
metrics_parsimonious= metrics_df.set_index(pd.Index(['L2LR','L1LR','L1SVM','L2SVM','NN','LGB']))
print(metrics_parsimonious)


# L2LR coefficients plots with error bars
df_coef_L2['coef_abs']=df_coef_L2['coef_'].abs()
LRresult = pd.merge(df_coef_L2.sort_values(by=['coef_abs'],ascending=False), explain_Variables[['Variable','Label']], how='left', on=['Variable'])
LRresult = LRresult.sort_values(ascending=False, by='coef_')
plt.errorbar(LRresult['Variable'], LRresult['coef_'].values, yerr=(1.96 * LRresult['standard_errors']),fmt='.k')
plt.xticks(rotation = 90)
plt.show()

# L2LR coefficients table with variable descriptions added from explain_Variables file
name_ = stat_test(X, X[col_y])
LRresult = LRresult.sort_values(ascending=False, by='coef_abs')
result_table = LRresult.merge(name_, on='Variable').drop(['coef_abs'],axis=1)
result_table

