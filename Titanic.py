import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as prf




#The features of the datasets will be the coloum name in data table. 
#'col_name' cointains the names of the features of the dataset of training data
#'test_col' cointains the names of the feature in test dataset
#'expected_col' cointains the name of the features that are to be predicted from 'test_col', here 'survival'

col_name = ['id', 'survival', 'p_class', 'name', 'sex', 'age', 'sib_sp', 'par_ch', 'ticket_no', 'fare', 'cabin_no', 'embarked']
test_col = ['id', 'p_class', 'name', 'sex', 'age', 'sib_sp', 'par_ch', 'ticket_no', 'fare', 'cabin_no', 'embarked']
expected_col =['id', 'survival']




#The file 'test.csv' cointains the data that should be used to train the hypothesis.
#'test.csv' cointains the data on which the hypothesis is to be tested. 
#The expected prediction, ie. 'survival', on dataset 'test.csv' is given in 'gender_submission.csv'

train_df = pd.read_csv('Data/train.csv', delimiter = ',', names = col_name, quotechar='"', skiprows = [0])
test_df = pd.read_csv('Data/test.csv', delimiter = ',', names = test_col, quotechar='"', skiprows = [0])
expected_df = pd.read_csv('Data/gender_submission.csv', delimiter = ',', names = expected_col, skiprows = [0]) 




#The train(train_df) dataset is divided into two parts
#The first part(train_outcome) cointains the values of the features that hypothesis should predict
#The secound part(train_data) cointains the values of the features on the basis of which hypothesis predicts.
#Simillary the required features of the test dataset are extracted where data wh=ith age is 'NaN' is ignored
#From Dummies variable we get two coloum, one which shows 1 if 'male' other coloum shows 1 if 'female'.
#We choose the former. This is done because logistic regression classifies can only optimize numerical inputs.

train_data = pd.DataFrame()
test_data = pd.DataFrame()

train_df = train_df.dropna()
train_data['p_class'] = train_df['p_class']
train_data['age'] = train_df['age']
train_data['sex'] = pd.get_dummies(train_df['sex'])['male']
train_outcome = train_df[ 'survival']

test_data['p_class'] = test_df['p_class']
test_data['age'] = test_df['age']
test_data['sex'] = pd.get_dummies(test_df['sex'])['male']




#There are data in testset which does not have age.
#For these datas, the age is assumed to be the median of age with same 'p_class' and 'sex'

for data_sex in [0 , 1]:
    #Sub-dataset with particular 'sex' is made
    train_data_gender = train_data[train_data['sex'] == data_sex]
    test_data_gender = test_data[test_data['sex'] == data_sex]
    for data_class in [1, 2 , 3]:
        #Sub-dataset with particular 'sex' and 'p_class' is made
        train_data_class = train_data_gender[train_data_gender['p_class'] == data_class]
        test_data_class = test_data_gender[test_data_gender['p_class'] == data_class].copy()
        
        #Median of age with the particular 'p_class' and 'sex' is computed
        missing_age = np.int(np.median(train_data_class['age']))
        
        #The data with missing 'age' is replaced with the computed age
        test_data_class['age'] = test_data_class['age'].fillna(missing_age)
        test_data[(test_data['sex'] == data_sex) & (test_data_gender['p_class'] == data_class)] = test_data_class




#The Feature are scaled inorder to avoid dominance of a type of data over others while training the logistic regression
#Here MinMaxScalar is used, which converts x to (x - xmin)/(xmax - xmin), where xmin and xmax are of training set

scaler = MinMaxScaler()
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)




#The data is being trained with various values of Regularization Strength and number of iteration.
#The Precision, Recall and F values for all the combination is stored in 'data_outcome'

##quality_features = ['C', 'Iteration', 'Precision', 'Recall', 'F1']
##hypothesis_quality = pd.DataFrame(data = None, columns = quality_features, dtype = float)
##
##predicted_df = pd.DataFrame(data = expected_df['id'])
##
##for c in [0.01, 0.3, 1.0 , 3.0, 10.0]:
##    for max_it in [100, 500, 1000, 2000]:
##        #The hypothesis was chosen and trained
##        hypothesis = LogisticRegression(C=c, max_iter = max_it, solver='lbfgs') 
##        hypothesis.fit(train_data, train_outcome)
##        
##        #The hypothesis was used to predicted and  values are stored in "predicted_df['survival']"
##        predicted_Series = pd.Series(data = hypothesis.predict(test_data), dtype = bool)
##        predicted_df['survival'] = predicted_Series
##        
##        #Precision, Recall, and F_Scores are computed and stored in 'hypothesis_quality'
##        #Along with the above values their Regularization Strength and number of iteration perfored is stored.
##        True_Positive = sum((expected_df['survival'] == 1) & (predicted_df['survival'] == 1))
##        False_Positive = sum((expected_df['survival'] == 0) & (predicted_df['survival'] == 1))
##        False_Negative = sum((expected_df['survival'] == 0) & (predicted_df['survival'] == 0))
##        
##        precision = True_Positive / (True_Positive + False_Positive)
##        recall = True_Positive / (True_Positive + False_Negative)
##        F_Score = (precision * (recall*2.0))/(recall + precision)
##        
##        hypothesis_quality.loc[hypothesis_quality.size/5] = [c, max_it , precision, recall, F_Score]




#The optimum value found for Regularization Strength is 1 and for number of iteration is 100.
#So the hypothesis is trained.
hypothesis = LogisticRegression(solver='lbfgs') 
hypothesis.fit(train_data, train_outcome)

#The values are used on test dataset to predict the survival of people
predicted_df = pd.DataFrame(data = expected_df['id'])
predicted_Series = pd.Series(data = hypothesis.predict(test_data), dtype = bool)
predicted_df['survival'] = predicted_Series

#The acuracy rate is competed against athe hypothesis where everyone is assumed to be dead
#The 'defult_accuracy' is the acurracy of the hypothesis which predicts everyone to be dead
#The 'predicted_accuracy' is he acurracy of the hypothesis which is trained using logistic regression.
no_of_prediction, no_of_features = predicted_df.shape

defult_accuracy = 1 - (sum(expected_df['survival'])/no_of_prediction)
predicted_accuracy = sum((predicted_df['survival'] == expected_df['survival']))/no_of_prediction
