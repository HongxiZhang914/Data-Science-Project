import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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
    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    # Now transform xtrain and xtest
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    return Xtrain,Xtest,ytrain,ytest,x,y

def neural_network(hidden_layer, b_size, iter, random_state,xtrain,xtest,ytrain,ytest):
    # fit/train the model. Check batch size.
    MLP = MLPClassifier(hidden_layer_sizes=(hidden_layer), batch_size=b_size, max_iter=iter, random_state=random_state)
    MLP.fit(xtrain,ytrain)
    # make Predictions
    ypred = MLP.predict(xtest)
    confusion_matrix(ytest, ypred)
    # check accuracy of the model
    print("Accuracy for the model is: ", accuracy_score(ytest, ypred))
    # Plotting loss curve
    loss_values = MLP.loss_curve_
    try:
        title = input("Please input the title for the chart: ")
        xlabel = input("Please enter the x label: ")
        ylabel = input("Please enter y lable: ")
    except:
        print("Please enter valid words")
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()
    return MLP
    
def grid_search(MLP,param,cv,x,y):
    # create a grid search
    grid = GridSearchCV(MLP, param, cv=cv, scoring='accuracy')
    grid.fit(x, y)
    print("The best batch size, hidden layer size and iteration is: ",grid.best_params_)    
    print("The best score can run is: ",grid.best_score_)
    
    
def main():
    # load the data using the pandas `read_csv()` function. 
    data = pd.read_csv('Admission.csv')
    # Converting the target variable into a categorical variable
    data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
    # Dropping columns
    data = data.drop(['Serial_No'], axis=1)
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    data = pd.get_dummies(data, columns=['University_Rating','Research'])
    Xtrain,Xtest,ytrain,ytest,x,y = data_partrition(data,0.2,123,'Admit_Chance')
    MLP = neural_network(3,50,100,123,Xtrain,Xtest,ytrain,ytest)
    # we will try different values for hyperparemeters
    params = {'batch_size':[20, 30, 40, 50],
            'hidden_layer_sizes':[(2,),(3,),(3,2)],
            'max_iter':[50, 70, 100]}
    grid_search(MLP,params,10,x,y)
    
if __name__ =="__main__":
    main()