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
    
    Partner_age_at_baseline_years = input("Enter partner age at baseline (years): ")# for example: 30
    
    Male_BMI_kg_m2 = input("Enter Combined male BMI (male and female reported): ")# for example: 27
    
    One_menstrual_cycle_of_attempt_time_at_study_entry = input("Enter 1 if One menstrual cycle of attempt time at study entry, otherwise enter 0: ")
    
    Partner_current_smoke__yes_regular_basis = input("Enter 1 if the answer to question (Partner current smoke) is “Yes, regular basis” , otherwise enter 0: ")
    
    improve_pregnancy_chances = input("Enter 1 if trying methods to improve pregnancy chances, otherwise enter 0: ")
    
    Year_since_last_pregnancy__0 = input("Enter 1 if there is 0 years since last pregnancy, otherwise enter 0: ")
    
    Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__no_never_tried_before = input("Enter 1 if the answer to question (Previously tried to conceive for ≥12 months) is “no, never tried before” , otherwise enter 0: ")
    
    Year_since_last_pregnancy__1_2 = input("Enter 1 if there are 1-2 years since last pregnancy, otherwise enter 0: ")
    
    Partner_current_smoke__yes_occasionally = input("Enter 1 if the answer to question (Partner current smoke) is “Yes, occasionally” , otherwise enter 0: ")
    
    Year_since_last_pregnancy__5 = input("Enter 1 if there are >= 5 years since last pregnancy, otherwise enter 0: ")
    
    Year_since_last_pregnancy__3_4 = input("Enter 1 if there are 3-4 years since last pregnancy, otherwise enter 0: ")
    
    Female_education_years = input("Enter Female education (years): ")# for example: 16
    
    Tap_water_freq_per_week = input("Enter Tap water freq per week: ")
    
    Intercourse_frequency = input("Enter Intercourse frequency (times/wk): ")
    
    Regular_period_on_its_own__No_irregular = input("Enter 1 if the period is irregular , otherwise enter 0: ")
    
    menstrual_cycle_been_regular_without_hormonal_BC__No_irregular = input("Enter 1 if menstrual cycle has been irregular in past 2 years , otherwise enter 0: ")
    
    Visited_doc_difficulty_getting_pregnant = input("Enter 1 if Visited doc for difficulty getting pregnant , otherwise enter 0: ")
    
    TTP_exit = input("Enter Cycles of attempt time at study exit: ")
    
    pregnant = input("Enter 1 if Pregnant including viable pregnancy and pregnancy loss , otherwise enter 0: ")
    
    Regular_period_on_its_own__Cannot_say = input("Enter 1 if it cannot be said that the period become regular on its own, because they were on hormones ; otherwise enter 0: ")
    
    menstrual_cycle_been_regular_without_hormonal_BC__Cannot_say = input("Enter 1 if it cannot be said that menstrual cycle has been regular in past 2 years, because they were on hormones ; otherwise enter 0: ")
    

    
    time_variable = float(TTP_exit) - float(One_menstrual_cycle_of_attempt_time_at_study_entry)
    
    
    variable_dict = {
        'bmi': (float(Female_BMI_kg_m2)-27.303669)/7.053749,
        'male_bmi_mf': (float(Male_BMI_kg_m2)-27.906815)/5.500748,
        'b_improvechances' : (float(improve_pregnancy_chances)-0.691265)/0.462027,
        'b_conteduc': (float(Female_education_years)-15.910719)/1.315451,
        'b_tapwaterfreq': (float(Tap_water_freq_per_week)-28.431648)/100.444105,
        'b_breastfeedever': (float(Ever_breastfed_an_infant)-0.309944)/0.462526,
        'b_intfreqc': (float(Intercourse_frequency)-2.143806)/1.599870,
        'pregsupp': (float(Daily_use_of_multivitamins_or_folic_acid)-0.820953)/0.383438,
        'b_yearssincelastpregcat_3.0' : (float(Year_since_last_pregnancy__1_2)-0.168885)/0.374695,
        'b_regularperiod_2.0' : (float(Regular_period_on_its_own__No_irregular)-0.207113)/0.405287,
        'b_nohormbcreg_2.0' : (float(menstrual_cycle_been_regular_without_hormonal_BC__No_irregular)-0.148318)/0.355458,
        'b_smokepartner_1.0': (float(Partner_current_smoke__yes_regular_basis)-0.090249)/0.286573,
        'b_trypregnant_1.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__yes)-0.067505)/0.250926,
        'b_trypregnantdoc': (float(Visited_doc_difficulty_getting_pregnant)-0.096056)/0.294704,
        'b_partnerage' : (float(Partner_age_at_baseline_years)-31.800871)/5.133954,
        'time_variable' : (float(time_variable)-4.083233)/3.558982,
        'pregnant' : (float(pregnant)-0.670215)/0.470192,
        'ttp_entry': (float(One_menstrual_cycle_of_attempt_time_at_study_entry)-0.579724)/0.493663,
        'b_regularperiod_3.0' : (float(Regular_period_on_its_own__Cannot_say)-0.118800)/0.323592,
        'b_nohormbcreg_3.0' : (float(menstrual_cycle_been_regular_without_hormonal_BC__Cannot_say)-0.331236)/0.470715,
        'b_trypregnant_3.0': (float(Previously_tried_to_conceive_for_greater_than_or_equal_to_12_months__no_never_tried_before)-0.398984)/0.489749,
        'b_smokepartner_2.0': (float(Partner_current_smoke__yes_occasionally)-0.042342)/0.201393,
        'b_yearssincelastpregcat_2.0' : (float(Year_since_last_pregnancy__0)-0.226470)/0.418597,
        'b_yearssincelastpregcat_4.0' : (float(Year_since_last_pregnancy__3_4)-0.048633)/0.215126,
        'b_yearssincelastpregcat_5.0' : (float(Year_since_last_pregnancy__5)-0.065570)/0.247559,
        
    }

    
    
    
    
    df = pd.DataFrame(columns=list(list(variable_dict.keys())))

    df = df.append(variable_dict, ignore_index=True)
    
    
    
    
    filename = 'cph'
    cph = pickle.load(open(filename, 'rb'))
    
    log_likelihood = cph.score(df, scoring_method='log_likelihood')

    print()
    print('The log_likelihood associated with this sample is: ' + str(log_likelihood))
    
