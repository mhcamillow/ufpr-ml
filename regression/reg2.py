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

base = pd.read_csv('usina72.csv', sep=',', encoding = 'ISO-8859-1', error_bad_lines=False)

features = base.iloc[:,[2,3,6,7,8,9]].values
labels_1 = base.iloc[:, 0]
labels_2 = base.iloc[:, 1]

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
labels_1 = scaler.fit_transform(labels_1.values.reshape(-1, 1))
labels_2 = scaler.fit_transform(labels_2.values.reshape(-1, 1))

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(features, labels_1, test_size = 0.50, random_state = 5)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(features, labels_2, test_size = 0.50, random_state = 5)

# regressor = LinearRegression()
# regressor = KNeighborsRegressor()
regressor = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.01)
# regressor = MLPRegressor(hidden_layer_sizes=(100, 200))
# regressor = DecisionTreeRegressor()
# regressor = RandomForestRegressor()
# regressor = GradientBoostingRegressor()
visualizer = ResidualsPlot(regressor)

visualizer.fit(x_train_2, y_train_2.ravel())
visualizer.score(x_test_2, y_test_2.ravel())
visualizer.poof()