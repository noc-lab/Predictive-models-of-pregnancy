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
#import seaborn as sns
import sklearn
import math
import pickle
import random
sklearn.__version__
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import os
os.listdir('/??????/')#DATA_PATH+

def pregnancy_probability():
    
    Menstrual_cycle_length_days = input("Enter Menstrual cycle length (days): ")# for example: 30

    Female_age_at_baseline_years = input("Enter Female age at baseline (years): ")# for example: 30

    Urbanization_of_residential_area__rural = input("Enter 1 if Urbanization of residential area is rural, otherwise enter 0:  ")

    Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes = input("Enter 1 if Previously tried to conceive for ≥12 months, otherwise enter 0:  ")

    One_menstrual_cycle_of_attempt_time_at_study_entry = input("Enter 1 if One menstrual cycle of attempt time at study entry, otherwise enter 0: ")

    Daily_use_of_multivitamins_or_folic_acid = input("Enter 1 if Daily use of multivitamins/folic acid, otherwise enter 0: ")

    Last_method_of_contraception__hormonal_IUD = input("Enter 1 if Last method of contraception: hormonal IUD, otherwise enter 0: ")

    Female_BMI_kg_m2 = input("Enter Female BMI (kg/m2): ")# for example: 27

    Ever_breastfed_an_infant = input("Enter 1 if Ever breastfed an infant, otherwise enter 0: ")

    Ever_been_pregnant = input("Enter 1 if Ever been pregnant, otherwise enter 0: ")

    Female_education_years = input("Enter Female education (years): ")# for example: 16

    Received_influenza_vaccine_in_the_past_year = input("Enter 1 if Received influenza vaccine in the past year, otherwise enter 0: ")

    Perceived_Stress_Scale_score = input("Enter Stress (Perceived Stress Scale score): ")# for example: 15

    Total_number_of_pregnancies = input("Enter Total number of pregnancies: ")# for example: 2

    Urbanization_of_residential_area__Canada = input("Enter 1 if Urbanization of residential area is Canada, otherwise enter 0:  ")

    Urbanization_of_residential_area__urban_cluster = input("Enter 1 if Urbanization of residential area is urban cluster, otherwise enter 0:  ")

    Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before = input("Enter 1 if the answer to question (Previously tried to conceive for ≥12 months) is “no, never tried before” (ref = “no, tried for < 12 months”), otherwise enter 0: ")

    
    variable_dict = {
            'ageatqstn' : Female_age_at_baseline_years,
            'b_everpregnant': Ever_been_pregnant,
            'b_gravid': Total_number_of_pregnancies,
            'b_fluvacc': Received_influenza_vaccine_in_the_past_year,
            'pregsupp': Daily_use_of_multivitamins_or_folic_acid,
            'pss_score': Perceived_Stress_Scale_score,
            'b_breastfeedever': Ever_breastfed_an_infant,
            'Cont_hormoneiud': Last_method_of_contraception__hormonal_IUD,
            'MenstruationCyclus': Menstrual_cycle_length_days,
            'ttp_entry': One_menstrual_cycle_of_attempt_time_at_study_entry,
            'bmi': Female_BMI_kg_m2,
            'b_conteduc': Female_education_years,
            'b_trypregnant_1.0': Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes,
            'ua_cat_1.0': Urbanization_of_residential_area__rural,
            'b_trypregnant_3.0': Previously_tried_to_conceive_for_greater_than_or_equal_to_months__no_never_tried_before,
            'ua_cat_2.0': Urbanization_of_residential_area__urban_cluster,
            'ua_cat_4.0': Urbanization_of_residential_area__Canada
    }
    
    
    
    df = pd.DataFrame(columns=list(list(variable_dict.keys())))

    df = df.append(variable_dict, ignore_index=True)

    my_seed = 2020
    prob = []
    pred = []
    my_seeds=range(my_seed, my_seed+5) # the random_state that controls the shuffling applied to the data before applying the split
    for seed in my_seeds:
        filename = f'clf_{seed}'
        clf = f'clf_{seed}'
        clf = pickle.load(open(filename, 'rb'))
        pred.append(clf.predict(df.values))
        prob.append(clf.predict_proba(df.values)[:,1])

    print()
    mean = sum(prob) / len(prob)
    print('The probability of pregnancy within 12 menstural cycles of pregnancy attempt time is: ' + str(round(mean[0],2)))
    