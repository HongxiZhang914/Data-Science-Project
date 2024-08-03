import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to scale the data using z-score
from sklearn.preprocessing import StandardScaler
#to split the dataset
from sklearn.model_selection import train_test_split
#import logistic regression
from sklearn.linear_model import LogisticRegression
#to build SVM model
from sklearn.svm import SVC
#Metrics to evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#for tuning the model
from sklearn.model_selection import GridSearchCV
#to ignore warnings
import warnings
warnings.filterwarnings("ignore")


def split_data(df,size,random_state, col):
    #Separating target variable and other variables
    Y= df[col]
    X= df.drop(columns = [col])
    #Scaling the data
    sc=StandardScaler()
    X_scaled=sc.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)
    #splitting the data
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=size,random_state=random_state,stratify=Y)
    return x_train,x_test,y_train,y_test

#creating metric function
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
def logistic_reg(x_train,y_train,x_test,y_test):
    #fitting logistic regression model
    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    #checking the performance on the training data
    y_pred_train = lg.predict(x_train)
    metrics_score(y_train, y_pred_train)
    #checking the performance on the test dataset
    y_pred_test = lg.predict(x_test)
    metrics_score(y_test, y_pred_test)
    
def SVM(kernel, x_train,y_train,x_test,y_test):
    #fitting SVM
    svm = SVC(kernel = kernel)
    model = svm.fit(X = x_train, y = y_train)
    y_pred_train_svm = model.predict(x_train)
    metrics_score(y_train, y_pred_train_svm)
    # Checking performance on the test data
    y_pred_test_svm = model.predict(x_test)
    metrics_score(y_test, y_pred_test_svm)

def main():
    #read the dataset
    df = pd.read_excel('HR_Employee_Attrition.xlsx')
    #dropping the columns
    df=df.drop(['EmployeeNumber','Over18','StandardHours'],axis=1)
    #Creating numerical columns
    num_cols=['DailyRate','Age','DistanceFromHome','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears',
            'YearsAtCompany','NumCompaniesWorked','HourlyRate',
            'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TrainingTimesLastYear']

    #Creating categorical variables
    cat_cols= ['Attrition','OverTime','BusinessTravel', 'Department','Education', 'EducationField','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance',
            'StockOptionLevel','Gender', 'PerformanceRating', 'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus','RelationshipSatisfaction']
    #creating histograms
    df[num_cols].hist(figsize=(14,14))
    plt.show()
    #Bivariate and Multivariate analysis
    for i in cat_cols:
        if i!='Attrition':
            (pd.crosstab(df[i],df['Attrition'],normalize='index')*100).plot(kind='bar',figsize=(8,4),stacked=True)
            plt.ylabel('Percentage Attrition %')
    #plotting the correlation between numerical variables
    plt.figure(figsize=(15,8))
    sns.heatmap(df[num_cols].corr(),annot=True, fmt='0.2f', cmap='YlGnBu')
    #creating list of dummy columns
    to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField','EnvironmentSatisfaction', 'Gender',  'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus' ]
    #creating dummy variables
    df = pd.get_dummies(data = df, columns= to_get_dummies_for, drop_first= True)
    #mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No':0}
    dict_attrition = {'Yes': 1, 'No': 0}
    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    x_train,x_test,y_train,y_test = split_data(df,0.2,1,'Attrition')
    logistic_reg(x_train,y_train,x_test,y_test)
    SVM('rbf', x_train,y_train,x_test,y_test)
    
if __name__ =="__main__":
    main()