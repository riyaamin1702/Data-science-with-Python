
#Visualisation 

from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly as py

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

le = LabelEncoder()
gender_encoded = le.fit_transform(df_features['Gender'])
days_encoded = le.fit_transform(df_features['DayofWeek'])
df_features = df_features.drop(['Month', 'DayofMonth', 'Year', 'TimeOfDay','Birthday', 'Lemmatized', 'POS_Tag'], axis=1)
df_features['Gender'] = gender_encoded
df_features['DayofWeek'] = days_encoded

df_train = pd.merge(df_train, df_features, left_index=True, right_index=True)
df_val = pd.merge(df_val, df_features, left_index=True, right_index=True)
df_test = pd.merge(df_test, df_features, left_index=True, right_index=True)

print('No of unique Beer by name:', df_train.BeerName.nunique())
print('No of unique Beer by ids:',df_train.BeerID.nunique())
## Counting number of users,reviewed the beer
print('No of unique users, reviewing the given beers: ',df_train.ReviewerID.nunique())
print('--'*50)
print('No of unique Beer by name in validation:', df_val.BeerName.nunique())
print('No of unique Beer by ids in validation:',df_val.BeerID.nunique())
print(df_train.rating.count())

#Figure 1: Total count of each beer rated by users
plt.figure(figsize=(20,4))
ax = sns.countplot(x ="BeerType", data =df_train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

#Figure 2: Most common beer rated by users
ax = df_train['BeerType'].value_counts()[:15].plot(kind = "bar")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Most Common Beer Type")

# Figure 3: Distribution of ABV
plt.figure(figsize=(8,4))
plt.hist("ABV", data = df_train, alpha = 0.6)
plt.title("Distribution of ABV")

plt.figure(figsize=(8,4))
plt.hist("ABV", data = df_val, alpha = 0.6)
plt.title("Distribution of ABV")


# Figure 4: Top 10 Beers by ABV
top_10_abv = df_train[['BeerName','BeerType','ABV']].\
    sort_values('ABV', ascending=False). \
    drop_duplicates('BeerName').\
    head(10).\
    sort_values('ABV', ascending=True)

# Combine brewery and beer name for readability
top_10_abv['combined_name'] = top_10_abv['BeerType'].str.\
  cat(top_10_abv['BeerName'], sep=' : ')

# Plot it
p = [go.Bar(x = top_10_abv['ABV'] / 100,
            y = top_10_abv['BeerName'],
            hoverinfo = 'x',
            text=top_10_abv['BeerType'],
            textposition = 'inside',
            orientation = 'h',
            opacity=0.7, 
            marker=dict(
                color='rgb(1, 87, 155)'))]

# Pieces of Flair
layout = go.Layout(title='Top 10 Strongest Beers by ABV',
                   xaxis=dict(title="ABV",
                              tickformat = "%",
                              hoverformat = '.2%'),
                   margin = dict(l = 220),
                   font=dict(family='Courier New, monospace',
                            color='dark gray'))

fig = go.Figure(data=p, layout=layout)

# Plot it
py.offline.iplot(fig)


# Figure 5: Correlation heatmap 
sns.heatmap( df_train[ [ 'BeerType', 'ABV', 'Gender', 'rating' ] ].corr(method = 'spearman'), center = 0,  vmin = -1, vmax = 1 )
plt.title( 'Spearman Correlation' )