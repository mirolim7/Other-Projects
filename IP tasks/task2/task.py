import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import prediction_error

df = pd.read_csv('kc_house_data.csv')
print('Just the head of dataset:')
print(df.head())
print('The shape of the dataset:')
print(df.shape)
print('The uniqueness of values in dataset:')
print(df.nunique())
print('General info about dataset:')
print(df.info())
print('General description of dataset:')
print(df.describe())

print('Distribution plot of target column("price"):')
print(sns.distplot(df['price']))


print(sns.scatterplot(range(df.shape[0]), np.sort(df.price.values)))

fig, ax = plt.subplots(figsize=(12, 10))
print('Correlation plot of dataset columns:')
print(sns.heatmap(df.corr(), linewidths=0.25, annot=True, ax=ax))


del df['id']
del df['date']


X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)


sc = [StandardScaler(), MinMaxScaler()]
md = [LinearRegression(), Ridge(), Lasso()]
ml = []
sl = []
msel = []
pcal = []

for m in md:
    m.fit(X_train, y_train)
    y_pred_train = m.predict(X_train)
    ml.append(str(m))
    sl.append('none')
    msel.append(metrics.mean_squared_error(y_train, y_pred_train))
    pcal.append('no')
    for s in sc:
        x = s.fit_transform(X_train)
        m.fit(x, y_train)
        y_pred_train = m.predict(x)
        ml.append(str(m))
        sl.append(str(s))
        msel.append(metrics.mean_squared_error(y_train, y_pred_train))
        pcal.append('no')
        
        steps = [
            ('scale', s),
            ('pca', PCA()),
            ('estimator', m)
        ]
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        
        ml.append(str(m))
        sl.append(str(s))
        msel.append(metrics.mean_squared_error(y_train, y_pred_train))
        pcal.append('yes')


pd.set_option('float_format', '{:f}'.format)

dict = {'model': ml, 'scaler': sl, 'mse': msel, 'pca': pcal} 
results = pd.DataFrame(dict, index=None)
print('Table of models, their metrics and the results:')
print(results)


lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
print('The best result was given by Linear Regression model, and mean squared error metric is:')
print(metrics.mean_squared_error(y_train, y_pred_train))

print('Sorted table for comparison')
print(results.sort_values(by='mse'))


y_pred = lr.predict(X_test)

print('The final plot for visualisation of machine learning model and the difference between original results and the predicted value:')
visualizer = prediction_error(lr, X_train, y_train, X_test, y_test)

