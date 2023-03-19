# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://user-images.githubusercontent.com/94169318/226172105-440bea62-15c2-40f2-b313-36f53c0c5fa5.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed by: Vidya Neela M
Register No. : 212221230120
```
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('my data1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
import pandas as pd
from sklearn.model_selection import train_test_split
# To scale
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X = df[['INPUT']].values
y = df[['OUTPUT']].values

X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=5000)
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)
model.evaluate(X_test1,y_test)
## new prediction
X_n1 = [[500]]
X_n1_1 = Scaler.transform(X_n1)
model.predict(X_n1_1)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/94169318/226155803-0ef7d4a0-ca4b-41b7-9328-7920c97dbf51.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/94169318/226155820-295c3b67-5611-499a-b5a2-06a4067982e9.png)


### Test Data Root Mean Squared Error


![image](https://user-images.githubusercontent.com/94169318/226172236-e3899b91-7cf9-41f5-92ec-939e9dcac0d7.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/94169318/226172214-dd85c57e-7e93-4b5e-a042-e4dce2921a0a.png)

## RESULT:
Thus a neural network regression model for the given dataset is written and executed successfully.
