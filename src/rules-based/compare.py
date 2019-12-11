# import packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# input files
NLP_RESULTS = './results/training_predictions'
FULL_DATASET = '../data/step1_fulldataset.csv'

#output files
OUT_FILE = './results/training_performance.txt'
file = open(OUT_FILE, 'w')

# load data
nlp_frame = pd.read_csv(NLP_RESULTS)
data_frame = pd.read_csv(FULL_DATASET)

# get performance results
def get_results(dataframe, goldstd, predicted):
    truepos = len(dataframe[(dataframe[goldstd] == 1) & (dataframe[predicted] == 1)])
    trueneg = len(dataframe[(dataframe[goldstd] == 0) & (dataframe[predicted] == 0)])
    falsepos = len(dataframe[(dataframe[goldstd] == 0) & (dataframe[predicted] == 1)])
    falseneg = len(dataframe[(dataframe[goldstd] == 1) & (dataframe[predicted] == 0)])

    sensitivity = truepos / (truepos + falseneg) * 100
    specificity = trueneg / (trueneg + falsepos) * 100
    ppv = truepos / (truepos + falsepos) * 100
    npv = trueneg / (trueneg + falseneg) * 100
    accuracy = (truepos + trueneg) / (truepos + trueneg + falsepos + falseneg) *100

    print ('Accuracy:\t\t\t' + str(accuracy))
    print ('Sensitivity:\t\t\t' + str(sensitivity))
    print ('Specificity:\t\t\t' + str(specificity))
    print ('Positive Predicted Value:\t' + str(ppv))
    print ('Negative Predicted Value:\t' + str(npv) + '\n')
            
    file.write ('Accuracy:\t\t\t' + str(accuracy) + '\n')
    file.write ('Sensitivity:\t\t\t' + str(sensitivity) + '\n')
    file.write ('Specificity:\t\t\t' + str(specificity) + '\n')
    file.write ('Positive Predicted Value:\t' + str(ppv) + '\n')
    file.write ('Negative Predicted Value:\t' + str(npv) + '\n')

# nlp
print ('NLP Performance: ')
file.write ('NLP Performance: \n')
get_results(nlp_frame, 'binary_adj_goldstd', 'predicted_class')

# medicare
def return_simpleicd(row):
    if row['countINPATIENT_afib'] >= 1 or row['countOUTPATIENT_afib'] >= 2:
        return 1
    else:
        return 0

combined_frame = nlp_frame.merge(data_frame, 'left', on='mrn')
combined_frame['simpleicd'] = combined_frame.apply(lambda row: return_simpleicd(row), axis=1)
print ('Medicare Performance: ')
file.write ('\nMedicare Performance: \n')
get_results(combined_frame, 'binary_adj_goldstd', 'simpleicd')

# kaiser
def return_kaiser(row):
    if (row['countINPATIENT_afib'] >= 2) or (row['countOUTPATIENT_afib'] >= 1 and row['ecg_afib_x'] == 1):
        return 1
    else:
        return 0

combined_frame['kaiser'] = combined_frame.apply(lambda row: return_kaiser(row), axis=1)
file.write ('\nKaiser Performance: \n')
print ('Kaiser Performance: ')
get_results(combined_frame, 'binary_adj_goldstd', 'kaiser')

# single reviewer

# logistic regression (limited)
X = pd.DataFrame(combined_frame, columns = ['agegrp1_x',
                                            'sex_x',
                                            'race_categ_x',
                                            'hispanic_x',
                                            'index_pay1_categ_x',
                                            'countINPATIENT_afib',
                                            'countOUTPATIENT_afib',
                                            'countall',
                                            'afibicd_primary_x',
                                            'index_year_x',
                                            'ami_x',
                                            'cad_x',
                                            'valve_x',
                                            'chf_x',
                                            'pvd_x',
                                            'cvd_x',
                                            'dementia_x',
                                            'pulmdz_x',
                                            'rheum_x',
                                            'ulcer_x',
                                            'liver_x',
                                            'dm_x',
                                            'renal_x',
                                            'ckd_x',
                                            'leuk_x',
                                            'lymph_x',
                                            'pulmhtn_x',
                                            'htn_x',
                                            'thyroid_x',
                                            'coag_x',
                                            'elec_x',
                                            'anemia_x',
                                            'cancer_x'
                                           ])
Y = pd.DataFrame(combined_frame, columns = ['binary_adj_goldstd'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
limited_regression_frame = pd.DataFrame({'y_test': np.array(y_test['binary_adj_goldstd']), 'y_pred': y_pred})
print ('Logistic Regression (Limited) Performance: ')
file.write ('\nLogistic Regression (Limited) Performance: \n')
get_results(limited_regression_frame, 'y_test', 'y_pred')

# logistic regression (expanded)
X = pd.DataFrame(combined_frame, columns = ['agegrp1_x',
                                            'sex_x',
                                            'race_categ_x',
                                            'hispanic_x',
                                            'index_pay1_categ_x',
                                            'countINPATIENT_afib',
                                            'countOUTPATIENT_afib',
                                            'countall',
                                            'afibicd_primary_x',
                                            'index_year_x',
                                            'ami_x',
                                            'cad_x',
                                            'valve_x',
                                            'chf_x',
                                            'pvd_x',
                                            'cvd_x',
                                            'dementia_x',
                                            'pulmdz_x',
                                            'rheum_x',
                                            'ulcer_x',
                                            'liver_x',
                                            'dm_x',
                                            'renal_x',
                                            'ckd_x',
                                            'leuk_x',
                                            'lymph_x',
                                            'pulmhtn_x',
                                            'htn_x',
                                            'thyroid_x',
                                            'coag_x',
                                            'elec_x',
                                            'anemia_x',
                                            'cancer_x',
                                            'ecg_afib_x',
                                            'dccv_cpt_binary_x',
                                            'ablate_cpt_binary_x'
                                           ])
Y = pd.DataFrame(combined_frame, columns = ['binary_adj_goldstd'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
expanded_regression_frame = pd.DataFrame({'y_test': np.array(y_test['binary_adj_goldstd']), 'y_pred': y_pred})
print ('Logistic Regression (Expanded) Performance: ')
file.write ('\nLogistic Regression (Expanded) Performance: \n')
get_results(expanded_regression_frame, 'y_test', 'y_pred')
            
file.close()
