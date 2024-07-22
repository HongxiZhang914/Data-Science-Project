#import all the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pickle

#Linear Regression
def LinearR(x_train,x_test,y_train,y_test):
    # train the model
    lrmodel = LinearRegression().fit(x_train,y_train)
    # make preditions on train set
    train_pred = lrmodel.predict(x_train)
    # evaluate the model using mean absolute error
    train_mae = mean_absolute_error(train_pred, y_train)
    print('Linear Regression Train error is', train_mae)
    # make predictions om test set
    ypred = lrmodel.predict(x_test)
    #evaluate the model
    test_mae = mean_absolute_error(ypred, y_test)
    print('Linear Regression Test error is', test_mae)
    
#Decision Tree Regression
def DecisionTree(x_train,x_test,y_train,y_test,MD,MF,RS):
    # create an instance of the class
    dt = DecisionTreeRegressor(max_depth=MD, max_features=MF, random_state=RS)
    # train the model
    dtmodel = dt.fit(x_train,y_train)
    # make predictions using the test set
    ytest_pred = dtmodel.predict(x_test)
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    print("Decision Tree test error is: ",test_mae)
    # make predictions on train set
    ytrain_pred = dtmodel.predict(x_train)
    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    print("Decision Tree Train error is: ",train_mae)
    return dtmodel
    
#Plot the Tree
def plot_the_tree(datamodel):
    tree.plot_tree(datamodel, feature_names=datamodel.feature_names_in_)
    tree.plot_tree(datamodel)
    plt.show()
    
#Random Forest Tree Model
def random_forest(x_train,x_test,y_train,y_test,estimator,crit):
    # create an instance of the model
    rf = RandomForestRegressor(n_estimators=estimator, criterion=crit)
    # train the model
    rfmodel = rf.fit(x_train,y_train)
    # make prediction on train set
    ytrain_pred = rfmodel.predict(x_train)
    # make predictions on the x_test values
    ytest_pred = rfmodel.predict(x_test)
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    print("Random Forest Test Error is: ",test_mae)
    return rfmodel
    
def main():
    #Read the file
    pd.set_option('display.max_columns', 50)
    df = pd.read_csv('final.csv')
    # seperate input features in x
    x = df.drop('price', axis=1)
    # store the target variable in y
    y = df['price']
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)
    #Train the model with different regression model
    LinearR(x_train,x_test,y_train,y_test)
    dtmodel = DecisionTree(x_train,x_test,y_train,y_test,3,10,567)
    rfmodel = random_forest(x_train, x_test, y_train, y_test,200,'absolute_error')
    # Save the trained model on the drive
    pickle.dump(rfmodel, open('RE_Model','wb'))
    plot_the_tree(dtmodel)
    
if __name__ =="__main__":
    main()



