
import pandas as pd
import numpy as np
from sklearn import preprocessing,linear_model
import sklearn

#### LODING DATA ####
print('_'*30);print('LOADING DATA');print('_'*30);
data = pd.read_csv('houses_to_rent.csv', sep = ',')
print(data.head())

data = data [['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance', 'furniture', 'rent amount']]
print(data.head())
#### PROCESS DATA ####
print('_'*30);print('PROCESSING DATA');print('_'*30);
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',','')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',','')))
le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform((data['furniture']))
print(data.head())

print('_'*30);print('CHECKING NULL DATA');print('_'*30);
print(data.isnull().sum())
#data = data.dropna()

#### HEAD ####
print('_'*30);print('HEAD');print('_'*30);
print(data.head())



#### SPLIT DATA ####
print('_'*30);print('SPLIT DATA');print('_'*30);
x = np.array(data.drop(['rent amount'], 1))
y = np.array(data['rent amount'])
print('X', x.shape)
print('Y', y.shape)

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y, test_size=0.3, random_state=10)

print('XTrain', xTrain.shape)
print('XTest', xTest.shape)


#### TRAINING DATA ####
print('_'*30);print(' TRAINING DATA');print('_'*30);
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy = model.score(xTest, yTest)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Accuracy:', round(accuracy*100,3),'%')


#### EVALUATION CHECK ####
print('_'*30);print(' EVALUATION CHECK');print('_'*30);
testVals = model.predict(xTest)
print(testVals.shape)
error = []
for i,testVal in enumerate(testVals):
    error.append(yTest[i]-testVal)
    print(f'Actual value:{yTest[i]} Prediction value:{int(testVal)} Error:{int(error[i])}')