import numpy as np
import pandas as pd
import csv
import sklearn
import math
import pickle
import warnings
warnings.filterwarnings("ignore")
sklearn.__version__
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def pregnancy_probability():
    
    Female_BMI_kg_m2 = input("Enter Female BMI (kg/m2): ")# for example: 27
    
    Daily_use_of_multivitamins_or_folic_acid = input("Enter 1 if Daily use of multivitamins/folic acid, otherwise enter 0: ")
    
    Ever_breastfed_an_infant = input("Enter 1 if Ever breastfed an infant, otherwise enter 0: ")
    
    Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes = input("Enter 1 if Previously tried to conceive for ≥12 months, otherwise enter 0:  ")
    
    Total_HEI_2010_score = input("Enter total Healthy Eating Index HEI-2010 score:")# for example: 65
    
    Partner_age_at_baseline_years = input("Enter partner age at baseline (years): ")# for example: 30
    
    Use_a_fertility_app = input("Enter 1 if there is no use a fertility app but they plan to use, otherwise enter 0: ")
    
    History_of_unplanned_pregnancy = input("Enter 1 if there is a history of unplanned pregnancy, otherwise enter 0: ")
    
    Male_BMI_kg_m2 = input("Enter Combined male BMI (male and female reported): ")# for example: 27
    
    One_menstrual_cycle_of_attempt_time_at_study_entry = input("Enter 1 if One menstrual cycle of attempt time at study entry, otherwise enter 0: ")
    
    Partner_current_smoke__yes_regular_basis = input("Enter 1 if the answer to question (Partner current smoke) is “Yes, regular basis” , otherwise enter 0: ")
    
    Female_age_at_baseline_years = input("Enter Female age at baseline (years): ")# for example: 30
    
    improve_pregnancy_chances = input("Enter 1 if trying methods to improve pregnancy chances, otherwise enter 0: ")
    
    Year_since_last_pregnancy__0 = input("Enter 1 if there is 0 years since last pregnancy, otherwise enter 0: ")
    
    History_of_subfertility_or_infertility= input("Enter 1 if there is a history of subfertility or infertility, otherwise enter 0: ")
    
    Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__no_never_tried_before = input("Enter 1 if the answer to question (Previously tried to conceive for ≥12 months) is “no, never tried before” , otherwise enter 0: ")
    
    Year_since_last_pregnancy__1_2 = input("Enter 1 if there are 1-2 years since last pregnancy, otherwise enter 0: ")
    
    Partner_current_smoke__yes_occasionally = input("Enter 1 if the answer to question (Partner current smoke) is “Yes, occasionally” , otherwise enter 0: ")
    
    Year_since_last_pregnancy__5 = input("Enter 1 if there are >= 5 years since last pregnancy, otherwise enter 0: ")
    
    Use_a_fertility_app_2 = input("Enter 1 if there is no use a fertility app and they do not plan to use, otherwise enter 0: ")
    
    Year_since_last_pregnancy__3_4 = input("Enter 1 if there are 3-4 years since last pregnancy, otherwise enter 0: ")
    
    
    
    variable_dict = {
        'ageatqstn' : (float(Female_age_at_baseline_years)-29.830265)/3.779226,
        'b_partnerage' : (float(Partner_age_at_baseline_years)-31.782221)/4.992545,
        'b_improvechances' : (float(improve_pregnancy_chances)-0.697929)/0.459222,
        'pregsupp': (float(Daily_use_of_multivitamins_or_folic_acid)-0.835731)/0.370573,
        'b_breastfeedever': (float(Ever_breastfed_an_infant)-0.303510)/0.459840,
        'b_unplannedpreg': (float(History_of_unplanned_pregnancy)-0.337745)/0.473009,
        'hxsubinfert' : (float(History_of_subfertility_or_infertility)-0.099827)/0.299813,
        'male_bmi_mf': (float(Male_BMI_kg_m2)-27.671244)/5.307756,
        'ttp_entry': (float(One_menstrual_cycle_of_attempt_time_at_study_entry)-0.579402)/0.493726,
        'bmi': (float(Female_BMI_kg_m2)-26.790671)/6.690414,
        'HEI2010_TOTAL_SCORE' : (float(Total_HEI_2010_score)-65.997764)/11.183027,
        'b_fertapp_3.0': (float(Use_a_fertility_app)-0.076237)/0.265415,
        'b_trypregnant_1.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes)-0.050345)/0.218688,
        'b_smokepartner_1.0': (float(Partner_current_smoke__yes_regular_basis)-0.078826)/0.269506,
        'b_yearssincelastpregcat_2.0' : (float(Year_since_last_pregnancy__0)-0.217491)/0.412599,
        'b_fertapp_2.0': (float(Use_a_fertility_app_2)-0.233602)/0.423183,
        'b_trypregnant_3.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__no_never_tried_before)-0.419160)/0.493493,
        'b_smokepartner_2.0': (float(Partner_current_smoke__yes_occasionally)-0.042002)/0.200623,
        'b_yearssincelastpregcat_3.0' : (float(Year_since_last_pregnancy__1_2)-0.172900)/0.378215,
        'b_yearssincelastpregcat_4.0' : (float(Year_since_last_pregnancy__3_4)-0.044879)/0.207069,
        'b_yearssincelastpregcat_5.0' : (float(Year_since_last_pregnancy__5)-0.060414)/0.238287,
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
    print('The probability of pregnancy within 6 menstural cycles of pregnancy attempt time is: ' + str(round(mean[0],2)))
    
