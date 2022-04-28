
#!/usr/bin/env python
# coding: utf-8
#Stacking Generalization

from numpy import mean
from numpy import std
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot


# get the dataset
def get_dataset(df_train, df_test, df_val):
    cleant = df_train.drop(['BeerName'], axis=1)
    cleanv = df_val.drop(['BeerName'], axis=1)
    cleanT = df_test.drop(['BeerName'], axis=1)

    X = cleant[['BeerID', 'ReviewerID', 'BeerType']]
    Xv = cleanv[['BeerID', 'ReviewerID', 'BeerType']]
    Xt = cleanT[['BeerID', 'ReviewerID', 'BeerType']]

    beertype = {x: i for i, x in enumerate(df_train['BeerType'].unique())}
    X.loc[:, 'BeerType'] = X['BeerType'].apply(lambda x: beertype[x])
    Xv.loc[:, 'BeerType'] = Xv['BeerType'].apply(lambda x: beertype[x])
    Xt.loc[:, 'BeerType'] = Xt['BeerType'].apply(lambda x: beertype[x])

    # X['ABV'] = df_features.loc[X.index, 'ABV']
    # Xv['ABV'] = df_features.loc[Xv.index, 'ABV']
    # Xt['ABV'] = df_features.loc[Xt.index, 'ABV']

    y = cleant[['rating']]
    yv = cleanv[['rating']]
    n = -1
    return X.values[:n], y.values.ravel()[:n], Xv.values[:n], yv.values.ravel()[:n], Xt.values

    #get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor()))
    #level0.append(('svm', SVR()))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model

    # get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['cart'] = DecisionTreeRegressor()
    models['stacking'] = get_stacking()
    # models['pls'] = PLSRegression(n_components=2)
    # models['svm'] = SVR()
    return models

    # evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores


if __name__ == "__main__":
    #with open('features-top5000.tsv', 'w') as fr:
     #   with open('features.tsv', 'r') as f:
      #      for i, ln in enumerate(f):
       #         if i == 5000:
        #            break
         #       fr.write(ln)

    df_train = pd.read_csv('train.tsv', sep='\t',
                    names=['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType', 'rating'])
    df_test = pd.read_csv('test.tsv', sep='\t', names=['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType'])
    df_val = pd.read_csv('val.tsv', sep='\t', names=['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType', 'rating'])
    df_features = pd.read_csv('features.tsv', sep='\t', names=['RowID', 'BrewerID', 'ABV', 'DayofWeek', 'Month',
                                                            'DayofMonth', 'Year', 'TimeOfDay', 'Gender',
                                                            'Birthday', 'Text', 'Lemmatized', 'POS_Tag'])

    df_train = df_train.set_index('RowID')
    df_test = df_test.set_index('RowID')
    df_val = df_val.set_index('RowID')
    df_features = df_features.set_index('RowID')
    
    #encode Gender and Dayofweek
    le = LabelEncoder()
    gender_encoded = le.fit_transform(df_features['Gender']) 
    days_encoded = le.fit_transform(df_features['DayofWeek'])
    df_features = df_features.drop(['Month', 'DayofMonth', 'Year', 'TimeOfDay','Birthday', 'Lemmatized', 'POS_Tag'], axis=1)
    df_features['Gender'] = gender_encoded
    df_features['DayofWeek'] = days_encoded
    
    #merge features in train, val and test
    df_train = pd.merge(df_train, df_features, left_index=True, right_index=True)
    df_val = pd.merge(df_val, df_features, left_index=True, right_index=True)
    df_test = pd.merge(df_test, df_features, left_index=True, right_index=True)

    X, y, Xv, yv, Xt = get_dataset(df_train, df_test, df_val)

    # create the base models
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    #pyplot.boxplot(results, labels=names, showmeans=True)
    #pyplot.show() #from the plot we see stacking doesnot perform well.

    model = get_stacking() #predict using stacking
    model.fit(X, y)
    scores = evaluate_model(model, Xv, yv)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    yhat = model.predict(Xt)
    # summarize prediction
    print('Predicted:', yhat)
    df_pred = pd.DataFrame({"RowID": df_test.index, "Prediction": np.round(100 * yhat) / 100})
    df_pred.to_csv("A3-3.tsv", sep = '\t', header=None,index=False)