#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

def data_partrition(df,size,state,*a):
    # Seperate the input features and target variable
    col = []
    for item in a:
        col.append(item)
    x = df.drop(columns = col,axis=1)
    y = df.drop(x,axis=1)
    # splitting the data in training and testing set
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=size, random_state=state)
    # scale the data using min-max scalar
    scale = MinMaxScaler()
    # Fit-transform on train data
    xtrain_scaled = scale.fit_transform(xtrain)
    xtest_scaled = scale.transform(xtest)
    return xtrain, xtest, ytrain, ytest,xtrain_scaled,xtest_scaled

def logistic_reg(ytrain, ytest,xtrain_scaled,xtest_scaled,threshold=None):
    lrmodel = LogisticRegression().fit(xtrain_scaled, ytrain)
    
    if threshold==None:
        ypred = lrmodel.predict(xtest_scaled)
        print("Accuracy for logistic regression is ",accuracy_score(ypred, ytest))
        return lrmodel,ypred
    else:
        yscores = lrmodel.predict_proba(xtest_scaled)[:, 1]
        ypred_with_threshold = (yscores >= threshold).astype(int)
        print("Accuracy for logictic regression model with threshold {} is {}".format(threshold,accuracy_score(ypred_with_threshold,ytest)))
        return lrmodel,ypred_with_threshold

def random_fore(xtrain, ytrain,xtest,ytest,estimators=100,leaf=5,features=None):
    rfmodel = RandomForestClassifier(n_estimators=estimators, min_samples_leaf=leaf, max_features=features)
    rfmodel.fit(xtrain, ytrain)
    ypred = rfmodel.predict(xtest)
    print("accuracy socre for random forest is: ",accuracy_score(ypred, ytest))
    return rfmodel

def cross_validation(model, xtrain, ytrain,split=5):
    # Set up a KFold cross-validation
    kfold = KFold(n_splits=split)
    # Use cross-validation to evaluate the model
    scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    # Print the accuracy scores for each fold
    print("Accuracy scores:", scores)
    # Print the mean accuracy and standard deviation of the model
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())

def main():
    # Import the data from 'credit.csv'
    df = pd.read_csv('credit.csv')
    # impute all missing values in all the features
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    # drop 'Loan_ID' variable from the data. We won't need it.
    df = df.drop('Loan_ID', axis=1)
    #change loan status variable type to int
    df['Loan_Status'] = df['Loan_Status'].eq('Y').mul(1)
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'])
    # saving this procewssed dataset
    df.to_csv('Processed_Credit_Dataset.csv', index=None)
    #data partition
    xtrain, xtest, ytrain, ytest,xtrain_scaled,xtest_scaled = data_partrition(df,0.2,123,'Loan_Status')
    lrmodel, prediction=logistic_reg(ytrain, ytest,xtrain_scaled,xtest_scaled,threshold=0.7)
    rfmodel= random_fore(xtrain, ytrain,xtest,ytest)
    print("logistic regression cross validation:")
    cross_validation(lrmodel, xtrain_scaled, ytrain)
    print("-----------------------------------------------------")
    print("Random Forest corss validation: ")
    cross_validation(rfmodel, xtrain_scaled, ytrain)
    
if __name__ =="__main__":
    main()
    