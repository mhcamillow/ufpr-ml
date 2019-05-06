import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def loadData():
    return pd.read_csv('usina72.csv', sep=',', encoding = 'ISO-8859-1', error_bad_lines=False)

def getFeaturesAndLabels(dataset, index):
    features = dataset.iloc[:,[2,3,6,7,8,9]].values
    labels = dataset.iloc[:, index]
    return Normalize(features, labels)

def Normalize(features, labels):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    labels = scaler.fit_transform(labels.values.reshape(-1, 1))
    return (features, labels)

def Split(features, labels):
    return train_test_split(features, labels, test_size = 0.50, random_state = 5)

def getRegressor(regressorName):
    if (regressorName == 'LinearRegression'):
        return LinearRegression()
    if (regressorName == 'KNeighborsRegressor'):
        return KNeighborsRegressor()
    if (regressorName == 'SVR'):
        return SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.01)
    if (regressorName == 'MLPRegressor'):
        return MLPRegressor(hidden_layer_sizes=(100, 200))
    if (regressorName == 'DecisionTreeRegressor'):
        return DecisionTreeRegressor()
    if (regressorName == 'RandomForestRegressor'):
        return RandomForestRegressor()
    if (regressorName == 'GradientBoostingRegressor'):
        return GradientBoostingRegressor()

# regressors = ['LinearRegression', 'KNeighborsRegressor', 'SVR', 'MLPRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']
regressors = ['LinearRegression']

for regressorName in regressors:
    for index in range(2):
        print('Running regressor ' + regressorName + ' on column ' + str(index))
        
        dataset = loadData()
        features, labels = getFeaturesAndLabels(dataset, index)
        x_train, x_test, y_train, y_test = Split(features, labels)

        my_title = regressorName + ' on column ' + str(index)
        visualizer = ResidualsPlot(getRegressor(regressorName), title=my_title)
        visualizer.fit(x_train, y_train.ravel())
        score = visualizer.score(x_test, y_test.ravel())
        visualizer.poof()

        print("Score: " + str(score))