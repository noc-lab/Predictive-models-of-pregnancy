#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2021

@author: Zahra Zad <zad@bu.edu>
@author: Taiyao Wang
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, plot_precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.feature_selection import RFE
from scipy import stats
import matplotlib.pyplot as plt
import argparse
from warnings import filterwarnings
filterwarnings('ignore')


def chi2_cols(y, x):
    '''
    input:
    y: 1-d binary label array
    x: 1-d binary feature array

    return:
    chi2 statistic and p-value
    '''
    y_list = y.astype(int).tolist()
    x_list = x.astype(int).tolist()
    freq = np.zeros([2, 2])
    for i in range(len(y_list)):
        if y_list[i] == 0 and x_list[i] == 0:
            freq[0, 0] += 1
        if y_list[i] == 1 and x_list[i] == 0:
            freq[1, 0] += 1
        if y_list[i] == 0 and x_list[i] == 1:
            freq[0, 1] += 1
        if y_list[i] == 1 and x_list[i] == 1:
            freq[1, 1] += 1
    y_0_sum = np.sum(freq[0, :])
    y_1_sum = np.sum(freq[1, :])
    x_0_sum = np.sum(freq[:, 0])
    x_1_sum = np.sum(freq[:, 1])
    total = y_0_sum+y_1_sum
    y_0_ratio = y_0_sum/total
    freq_ = np.zeros([2, 2])
    freq_[0, 0] = x_0_sum*y_0_ratio
    freq_[0, 1] = x_1_sum*y_0_ratio
    freq_[1, 0] = x_0_sum-freq_[0, 0]
    freq_[1, 1] = x_1_sum-freq_[0, 1]
    stat, p_value = stats.chisquare(freq, freq_, axis=None)
    return p_value


def stat_test(df, y):
    name = pd.DataFrame(df.columns, columns=['Variable'])
    df0 = df[y == 0]
    df1 = df[y == 1]
    pvalue = []
    y_corr = []
    for col in df.columns:
        if df[col].nunique() == 2:
            pvalue.append(chi2_cols(y, df[col]))
        else:
            pvalue.append(stats.ks_2samp(df0[col], df1[col]).pvalue)
        y_corr.append(df[col].corr(y))
    name['All_mean'] = df.mean().values
    name['y1_mean'] = df1.mean().values
    name['y0_mean'] = df0.mean().values
    name['All_std'] = df.std().values
    name['y1_std'] = df1.std().values
    name['y0_std'] = df0.std().values
    name['p-value'] = pvalue
    name['y_corr'] = y_corr
    return name.sort_values(by=['p-value'])


def df_ycorr(df, col_y):
    drop_cols = []
    for col in df.columns:
        if df[col].nunique() != 2:
            y_corr = round(df[col_y].corr(df[col]), 2)
            if (abs(y_corr) < 0.04):
                drop_cols.append(col)
    return drop_cols


def high_corr(df, thres=0.8):
    corr_matrix_raw = df.corr()
    corr_matrix = corr_matrix_raw.abs()
    high_corr_var_ = np.where(corr_matrix > thres)
    high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix_raw.iloc[x, y])
                     for x, y in zip(*high_corr_var_) if x != y and x < y]
    return high_corr_var


def df_drop(df_new, drop_cols):
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)


def my_train(X_train, y_train, model='LR', penalty='l1', cv=5, scoring='roc_auc', class_weight='balanced', seed=2020):
    if model == 'SVM':
        svc = LinearSVC(penalty=penalty, class_weight=class_weight,
                        dual=False, max_iter=5000)  # , tol=0.0001
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gsearch = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring)
    elif model == 'LGB':
        param_grid = {
            'bagging_fraction': [0.9],
            'nthread': [3],
            'num_leaves': range(6, 12, 2),
            'min_data_in_leaf': range(14, 26, 2),
            'learning_rate': [0.08, 0.10, 0.12, 0.14],
            'feature_fraction': [0.2, 0.3, 0.4, 0.5, 0.6]
        }
        lgb_estimator = lgb.LGBMClassifier(
            boosting_type='gbdt',  objective='binary', class_weight=class_weight, random_state=seed)
        gsearch = GridSearchCV(
            estimator=lgb_estimator, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=scoring)
    elif model == 'MLP':
        mlp = MLPClassifier(random_state=seed, tol=0.01)
        param_grid = {'hidden_layer_sizes': [
            (C, ), (round(C/2), 2), (round(C/4), 4)] for C in [32, 64, 128, 256, 512]}
        gsearch = GridSearchCV(mlp, param_grid, cv=cv, scoring=scoring)
    else:
        LR = LogisticRegression(
            penalty=penalty, class_weight=class_weight, solver='liblinear', random_state=seed)
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10]}
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring)
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    return clf


def cal_f1_scores(y, y_pred_score):
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    thresholds = sorted(set(thresholds))
    metrics_all = []
    for thresh in thresholds:
        y_pred = np.array((y_pred_score > thresh))
        metrics_all.append((thresh, auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(
            y, y_pred, average='macro'), f1_score(y, y_pred, average='weighted')))
    metrics_df = pd.DataFrame(metrics_all, columns=[
                              'thresh', 'AUC',  'micro_F1_score', 'macro_F1_score', 'weighted_F1_score'])
    return metrics_df.sort_values(by='weighted_F1_score', ascending=False).head(1)['thresh'].values[0]


def cal_f1_scores_te(y, y_pred_score, thresh):
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    y_pred = np.array((y_pred_score > thresh))
    if if_RFE == 0:
        metrics_all = [(thresh, auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(y, y_pred, average='macro'), f1_score(
            y, y_pred, average='weighted'), precision_score(y, y_pred, average='weighted'), recall_score(y, y_pred, average='weighted'))]
        metrics_df = pd.DataFrame(metrics_all, columns=[
                                  'thresh', 'AUC', 'micro_F1_score', 'macro_F1_score', 'weighted_F1_score', 'weighted_precision_score', 'weighted_recall_score'])
    else:
        metrics_all = [(thresh, auc(fpr, tpr), f1_score(y, y_pred, average='micro'), f1_score(
            y, y_pred, average='macro'), f1_score(y, y_pred, average='weighted'))]
        metrics_df = pd.DataFrame(metrics_all, columns=[
                                  'thresh', 'AUC',  'micro_F1_score', 'macro_F1_score', 'weighted_F1_score'])
    return metrics_df


def my_test(X_train, xtest, y_train, ytest, clf, target_names, report=False, model='LR'):
    if model == 'SVM':
        ytrain_pred_score = clf.decision_function(X_train)
    else:
        ytrain_pred_score = clf.predict_proba(X_train)[:, 1]
    thres_opt = cal_f1_scores(y_train, ytrain_pred_score)
    if model == 'SVM':
        ytest_pred_score = clf.decision_function(xtest)
    else:
        ytest_pred_score = clf.predict_proba(xtest)[:, 1]
    return cal_f1_scores_te(ytest, ytest_pred_score, thres_opt)


def tr_predict(df_new, col_y, target_names=['0', '1'], model='LR', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False):
    scaler = preprocessing.StandardScaler()
    y = df_new[col_y].values
    metrics_all = []
    my_seeds = range(2020, 2025)
    for seed in my_seeds:
        X = df_new.drop([col_y], axis=1).values
        name_cols = df_new.drop([col_y], axis=1).columns.values
        X = scaler.fit_transform(X)
        X_train, xtest, y_train, ytest = train_test_split(
            X, y, stratify=y, test_size=test_size,  random_state=seed)
        clf = my_train(X_train, y_train, model=model, penalty=penalty,
                       cv=cv_folds, scoring=scoring, class_weight='balanced', seed=seed)
        metrics_all.append(my_test(X_train, xtest, y_train,
                                   ytest, clf, target_names, report=report, model=model))
    metrics_df = pd.concat(metrics_all)
    if if_RFE == 0:
        metrics_df = metrics_df[['AUC', 'micro_F1_score', 'weighted_F1_score', 'weighted_precision_score',
                                 'weighted_recall_score']].describe().T[['mean', 'std']].stack().to_frame().T
    else:
        metrics_df = metrics_df[['AUC', 'micro_F1_score', 'weighted_F1_score']].describe(
        ).T[['mean', 'std']].stack().to_frame().T
    # df_coef_ based on model type
    if model == 'LGB':
        df_coef_ = pd.DataFrame(list(zip(name_cols, np.round(
            clf.feature_importances_, 2))), columns=['Variable', 'coef_'])
        fig, ax = plt.subplots()
        lgb.plot_importance(clf, ax=ax, max_num_features=100)
        plt.title("Light GBM Feature Importance")
    elif model == 'MLP':
        df_coef_ = pd.DataFrame(list(zip(name_cols, np.round(
            clf.coefs_[0][:, 0], 2))), columns=['Variable', 'coef_'])
    else:
        df_coef_ = pd.DataFrame(
            list(zip(name_cols, np.round(clf.coef_[0], 5))), columns=['Variable', 'coef_'])
        if ((model == 'LR') & (if_RFE == 0)):
            # plot_precision_recall_curve
            disp = plot_precision_recall_curve(clf, xtest, ytest)
            disp.ax_.axis(ymin=0, ymax=1)
            disp.ax_.set_title('2-class Precision-Recall curve')
            # calculate standard_errors
            predProbs = clf.predict_proba(X_train)
            X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
            V = np.product(predProbs, axis=1)
            covLogit = np.linalg.pinv(np.dot((X_design.T * V), X_design))
            standard_errors = np.sqrt(np.diag(covLogit))
            df_coef_['standard_errors'] = standard_errors[1:]
    df_coef_['coef_abs'] = df_coef_['coef_'].abs()
    if ((model == 'LR') & (if_RFE == 0)):  # because of the column satandard_errors
        # , scaler
        return df_coef_.sort_values('coef_abs', ascending=False)[['coef_', 'Variable', 'standard_errors']], metrics_df
    else:
        # , scaler
        return df_coef_.sort_values('coef_abs', ascending=False)[['coef_', 'Variable']], metrics_df


def my_RFE(df_new, col_y, my_range=range(1, 11), my_C_range=[0.01, 0.1, 1], my_penalty='l1', class_weight='balanced', solver='liblinear'):
    metric_all_rfe = []
    Xraw = df_new.drop(col_y, axis=1).values
    y = df_new[col_y].astype(int)
    for my_C in my_C_range:
        for n_select in my_range:
            scaler = preprocessing.StandardScaler()
            X = scaler.fit_transform(Xraw)
            clf = LogisticRegression(
                C=my_C, penalty=my_penalty, tol=0.01, class_weight=class_weight, solver=solver)
            rfe = RFE(clf, n_select, step=1)
            rfe.fit(X, y.ravel())
            X = df_new.drop(names[rfe.ranking_ > 1], axis=1)
            df_coef_RFE, metric_df_RFE = tr_predict(X, col_y, target_names=[
                                                    '0', '1'], model='LR', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
            metric_all_rfe.append(
                [my_C, n_select]+metric_df_RFE.values.tolist()[0])
    metric_all_rfe = pd.DataFrame(metric_all_rfe, columns=[
                                  'my_C', 'n_select', 'AUC-mean', 'AUC-std', 'micro_F1_score-mean', 'micro_F1_score-std', 'weighted_F1_score-mean', 'weighted_F1_score-std'])
    metric_all_rfe['AUC_'] = metric_all_rfe['AUC-mean'] - \
        metric_all_rfe['AUC-std']
    scaler = preprocessing.StandardScaler()  # MinMaxScaler
    X = scaler.fit_transform(Xraw)
    clf = LogisticRegression(C=metric_all_rfe.loc[metric_all_rfe['AUC_'].idxmax(
    ), 'my_C'], penalty=my_penalty, tol=0.01, class_weight='balanced', solver='liblinear')
    rfe = RFE(
        clf, metric_all_rfe.loc[metric_all_rfe['AUC_'].idxmax(), 'n_select'], step=1)
    rfe.fit(X, y.ravel())
    X = df_new.drop(names[rfe.ranking_ > 1], axis=1)
    return X


#load dataset###############################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', help='the path of dataset in csv format')
    args = parser.parse_args()

    df_new = pd.read_csv(args.dataset_path)

    df_new = df_new.drop(
        df_new.columns[df_new.columns.isin(['newid'])], axis=1)
    df_new = df_new.loc[df_new['ttp_entry'] < 2, :].reset_index(drop=True)
    which_month = 12  # 7
    df_new.loc[(df_new['TTP_exit'] >= which_month), ['pregnant']] = 0
    df_new = df_new[(df_new['TTP_exit'] >= which_month) | (
        (df_new['TTP_exit'] < which_month) & (df_new['pregnant'] == 1))]
    df_new = df_new.reset_index(drop=True)
    df_new['b_livebirths'] = round(df_new['b_livebirths'])
    for col in ['b_birthorder', 'b_livebirths']:
        df_new.loc[df_new[col] > 3, col] = 3
    for col in ['b_csectiontotal']:
        df_new.loc[df_new[col] > 2, col] = 2
    drop_cols = ['TTP_exit', 'b_finisheddate_year']
    df_new = df_drop(df_new, drop_cols)

    #Statistical Feature Selection##############
    ##drop variables with low std###############
    col_y = 'pregnant'
    df_std = df_new.std()
    drop_cols = df_std[df_std < 0.0001].index.values
    df_new = df_drop(df_new, drop_cols)

    ##drop variables with high p-value##########
    result = stat_test(df_new, df_new[col_y])
    drop_cols = result.loc[result['p-value'] > 0.05, 'Variable'].values
    df_new = df_drop(df_new, drop_cols)

    # drop continuous variables with low correlation with y
    drop_cols = df_ycorr(df_new, col_y)
    df_new = df_drop(df_new, drop_cols)

    ##keep one var among high correlated vars####
    high_corr_features = high_corr(df_new, thres=0.8)
    print(f'high_corr_features = {high_corr_features}')
    # select one to drop among each pair in high_corr_list
    drop_cols = ['b_livebirths', 'hxinfert']
    df_new = df_drop(df_new, drop_cols)

    #full models################################
    y = df_new[col_y].astype(int)
    names = df_new.drop(col_y, axis=1).columns
    if_RFE = 0
    df_coef_L2, metrics_df_L2 = tr_predict(df_new, col_y, target_names=[
                                           '0', '1'], model='LR', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df_bf = metrics_df_L2
    df_coef_L1, metrics_df_L1 = tr_predict(df_new, col_y, target_names=[
                                           '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df_bf = pd.concat([metrics_df_bf, metrics_df_L1])
    df_coef_svmL1, metrics_df_svmL1 = tr_predict(df_new, col_y, target_names=[
                                                 '0', '1'], model='SVM', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df_bf = pd.concat([metrics_df_bf, metrics_df_svmL1])
    df_coef_svmL2, metrics_df_svmL2 = tr_predict(df_new, col_y, target_names=[
                                                 '0', '1'], model='SVM', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df_bf = pd.concat([metrics_df_bf, metrics_df_svmL2])
    df_coef_mlp, metrics_df_mlp = tr_predict(df_new, col_y, target_names=[
                                             '0', '1'], model='MLP', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df_bf = pd.concat([metrics_df_bf, metrics_df_mlp])
    df_coef_LGB, metrics_df_LGB = tr_predict(df_new, col_y, target_names=[
                                             '0', '1'], model='LGB', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df_bf = pd.concat([metrics_df_bf, metrics_df_LGB])
    metrics_full = metrics_df_bf.set_index(
        pd.Index(['L2LR', 'L1LR', 'L1SVM', 'L2SVM', 'NN', 'LGB']))
    print(metrics_full)

    #sparse models##############################
    if_RFE = 1  # Featture selection by Recursive Feature Elimination
    my_range = range(1, len(df_new.columns))
    X = my_RFE(df_new, col_y, my_range=my_range, my_C_range=[
               0.01, 0.1, 1], my_penalty='l1', class_weight='balanced', solver='liblinear')
    if_RFE = 0
    df_coef_L2, metrics_df_L2 = tr_predict(X, col_y, target_names=[
                                           '0', '1'], model='LR', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = metrics_df_L2
    df_coef_L1, metrics_df_L1 = tr_predict(X, col_y, target_names=[
                                           '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_L1])
    df_coef_svmL1, metrics_df_svmL1 = tr_predict(X, col_y, target_names=[
                                                 '0', '1'], model='SVM', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_svmL1])
    df_coef_svmL2, metrics_df_svmL2 = tr_predict(X, col_y, target_names=[
                                                 '0', '1'], model='SVM', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_svmL2])
    df_coef_mlp, metrics_df_mlp = tr_predict(X, col_y, target_names=[
                                             '0', '1'], model='MLP', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_mlp])
    df_coef_LGB, metrics_df_LGB = tr_predict(X, col_y, target_names=[
                                             '0', '1'], model='LGB', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_LGB])
    metrics_sparse = metrics_df.set_index(
        pd.Index(['L2LR', 'L1LR', 'L1SVM', 'L2SVM', 'NN', 'LGB']))
    print(metrics_sparse)

    #parsimonious models########################
    if_RFE = 1
    my_range = range(1, 3)
    X = my_RFE(df_new, col_y, my_range=my_range, my_C_range=[
               0.01, 0.1, 1], my_penalty='l1', class_weight='balanced', solver='liblinear')
    if_RFE = 0
    df_coef_L2, metrics_df_L2 = tr_predict(X, col_y, target_names=[
                                           '0', '1'], model='LR', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = metrics_df_L2
    df_coef_L1, metrics_df_L1 = tr_predict(X, col_y, target_names=[
                                           '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_L1])
    df_coef_svmL1, metrics_df_svmL1 = tr_predict(X, col_y, target_names=[
                                                 '0', '1'], model='SVM', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_svmL1])
    df_coef_svmL2, metrics_df_svmL2 = tr_predict(X, col_y, target_names=[
                                                 '0', '1'], model='SVM', penalty='l2', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_svmL2])
    df_coef_mlp, metrics_df_mlp = tr_predict(X, col_y, target_names=[
                                             '0', '1'], model='MLP', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_mlp])
    df_coef_LGB, metrics_df_LGB = tr_predict(X, col_y, target_names=[
                                             '0', '1'], model='LGB', cv_folds=5, scoring='roc_auc', test_size=0.2, report=False)
    metrics_df = pd.concat([metrics_df, metrics_df_LGB])
    metrics_parsimonious = metrics_df.set_index(
        pd.Index(['L2LR', 'L1LR', 'L1SVM', 'L2SVM', 'NN', 'LGB']))
    print(metrics_parsimonious)

    #L2LR coefficients plots####################
    df_coef_L2['coef_abs'] = df_coef_L2['coef_'].abs()
    LRresult = df_coef_L2.sort_values(ascending=False, by='coef_')
    plt.errorbar(LRresult['Variable'], LRresult['coef_'].values, yerr=(
        1.96 * LRresult['standard_errors']), fmt='.k')
    plt.xticks(rotation=90)
    plt.show()
    #L2LR coefficients table####################
    name_ = stat_test(X, X[col_y])
    result_table = LRresult.merge(
        name_, on='Variable').drop(['coef_abs'], axis=1)
    print(result_table)
    #end########################################
