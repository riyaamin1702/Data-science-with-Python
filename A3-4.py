#!/usr/bin/env python
# coding: utf-8
#Ensemble Learning with blending

from os import sep
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder


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


# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['cart'] = DecisionTreeRegressor()
    #     models['svm'] = SVR()
    #     models['stacking'] = get_stacking()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# fit the blending ensemble
def fit_ensemble(models, X, y, Xv, yv, cv):
    meta_X = list()
    for train_index, val_index in cv.split(X):
        print("TRAIN:", len(train_index), "TEST:", len(val_index))
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # fit all models on the training set and predict on hold out set
        for name, model in models.items():
            # fit in training set
            model.fit(X_train, y_train)
            scores = evaluate_model(model, X_val, y_val)
            print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        break
    for name, model in models.items():
        # predict on hold out set
        yhat = model.predict(Xv)
        # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
        # store predictions as input for blending
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = np.hstack(meta_X)
    # define blending model
    blender = LinearRegression()
    # fit on predictions from base models
    blender.fit(meta_X, yv)
    return blender


# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
    # make predictions with base models
    meta_X = list()
    for _, model in models.items():
        # predict with base model
        yhat = model.predict(X_test)
        # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
        # store prediction
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = np.hstack(meta_X)
    # predict
    return blender.predict(meta_X)


if __name__ == "__main__":
    #with open('features-top5000.tsv', 'w') as fr:
    #    with open('features.tsv', 'r') as f:
    #        for i, ln in enumerate(f):
    #            if i == 5000:
    #                break
    #           fr.write(ln)

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
    # train the blending ensemble
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    blender = fit_ensemble(models, X, y, Xv, yv, cv)

    # ##  Evaluate on Val set
    yhat = predict_ensemble(models, blender, Xv)
    print("Super learner MAE", np.mean(abs(yv - yhat)))

    yhat = predict_ensemble(models, blender, Xt)
    # summarize prediction
    print('Predicted:', yhat)
    df_pred = pd.DataFrame({"RowID": df_test.index, "Prediction": np.round(100 * yhat) / 100})
    df_pred.to_csv("A3-4.tsv", sep= '\t', header=None, index=False)
