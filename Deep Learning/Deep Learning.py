import pandas as pd
#to scale the data using z-score
from sklearn.preprocessing import StandardScaler
#to split the dataset
from sklearn.model_selection import train_test_split
#Metrics to evaluate the model
#to ignore warnings
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

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

def neural_network(hidden_layer,dense, x_train,y_train,epochs,verbose):
    # set a fixed random seed for the model's weight initialization
    tf.keras.utils.set_random_seed(42)
    # 1. Create the model using the Sequential API
    layers = []
    for i in range(0,hidden_layer):
        layers.append(tf.keras.layers.Dense(dense[i]))
    model_1 = tf.keras.Sequential(layers)
    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), # binary since we are working with 2 clases (0 & 1)
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])
    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=epochs,verbose=verbose)
    print(model_1.evaluate(x_train, y_train))

def main():
    #read the dataset
    df = pd.read_csv('employee_attrition.csv')
    x_train,x_test,y_train,y_test = split_data(df,0.2,1,'Attrition')
    neural_network(2,[2,1],x_train=x_train,y_train=y_train,epochs=50,verbose=0)
    
if __name__ =="__main__":
    main()
    