![Ironhack logo](https://i.imgur.com/1QgrNNw.png)

# Cybercrime

This is my final project at Ironhack, which helped me get to the Final Hackshow on the 13th of December.

## Final Presentation

For the final presentation please check the following Google Slides: 
https://docs.google.com/presentation/d/113M_4-i2eDk3lb1LXbMqv33i7r_5MPepVMrfasY1NUQ/edit?usp=sharing

## Goal of the Project

My goal of this project is to inform the audience about the world cybercrime scene and, in particular, about credit card fraud. I've also had a shot at the fraud detection algorithms by training the Deep Learning Model.

## Deep Learning Model

### Data Sources:

* [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) - Credit Card Fraud

### Prerequisites

Installation of the necessary libraries

```
# Essensial libraries

import pandas as pd
import numpy as np

# Plotting libraries

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import SVG

# Scikit-Learn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Artificial Neural Networks

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import model_to_dot

# Additional

np.random.seed(123)
%matplotlib inline
```

### Plotting the correlation of the variables

It is a necessary step, which allows us to get some insights in the data

```
adjustment = np.zeros_like(correlation, dtype = np.bool)
adjustment[np.triu_indices_from(adjustment)] = True

fig, ax = plt.subplots(figsize = (18, 15))

cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(correlation, mask = adjustment, cmap = cmap, vmax = 0.3, center = 0,
            square = True, linewidths = 0.5, cbar_kws={"shrink": 0.5})
```

### Scaling the variable Amount

It is necessary to adjust this variable since the values of it are big. As well as dropping the Time column

```
df['Scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Amount'], axis = 1)

# Dropping the Time

df = df.drop(['Time'], axis = 1)
df.head()
```

### Assigning X and y and splitting into train and test sets

Our target variable is Class, which states if the card was stolen or not

```
X = df.iloc[:, df.columns != 'Class']
y = df.iloc[:, df.columns == 'Class']
```

```
# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```

### Training the Deep Learning Model

Deep Learning, in general, is a subset of Machine Learning, but there are numerous layers containing algorothms, each providing a different interpretation to the data it feeds on.

My decision was to go with Sequential classifier(model), which is simply a linear stack of layers. This means that every layer has one input and output. The output of oen layer is the input of another.

Initalizer was 'uniform', since I had an equal possibility of outcomes (fraud/not fraud)
Activation used for input and hidden layers was 'relu', because it performs well and for output it was 'sigmoid' for the sense of binary classification.

Thinking about compiling, I've used 'Adam' as an optimizer since it adjusts the learning rate during training, loss was 'binary crossentropy' since I had 2 target classes and 'accuracy' as metrics.

Batch size and epochs were chosen in order not to spend huge amount of computational resources.

```
classifier = Sequential()

# Input layer and the first hidden layer

classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Second hidden layer

classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN

classifier.fit(X_train, y_train, batch_size = 32, epochs = 15)
```

### Results

Getting the score

```
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
score = classifier.evaluate(X_test, y_test)
```

Confusion matrix

```
matrix = confusion_matrix(y_test, y_pred)
```

Summary

```
classifier.summary()
```

Test Data Accuracy

```
print('Test Data Accuracy: %0.4f' % accuracy_score(y_test, y_pred))
```


## World Globe

Working with this type of Visualization was a bit new for me, that is why I've adjusted only the data source without changing a lot of the source code. The data file was in JSON format, so I've imported it into Jupyter Notebook and have done the adjustments there.

The Globe visualization consists of two different parts: victims and aggressors

### Data Sources:

* [CYBERTHREAT REAL-TIME MAP](https://cybermap.kaspersky.com) - Victims
* [AWS Honeypot Attack Data](https://www.kaggle.com/casimian2000/aws-honeypot-attack-data) - Aggressors

### Preparing 'Victims' Data

```
# Impoting the libraries

import pandas as pd
import reverse_geocode as rg

# Importing the file

df = pd.read_json('example.json')

# Getting the data out of the column 1 in the lists

tags = df[1].apply(pd.Series)
tags = tags.rename(columns = lambda x: 'tag_' + str(x))

# Processing with latitude and longitude

tags_adj = tags.transpose()
data_1 = tags_adj[0]
latitude = data_1[0::3]
longitude = data_1[1::3]
latitude = latitude.reset_index(drop = True)
longitude = longitude.reset_index(drop = True)

# Creating a new dataframe

was = pd.DataFrame()
was['longitude'] = longitude
was['latitude'] = latitude

# Getting the country names

x = was['longitude'].tolist()
y = was['latitude'].tolist()
z = list(zip(y, x))

location = rg.search(z)
countries = []
for k, v in [(k, v) for x in location for (k, v,) in x.items()]:
    print(type(v))
    try:
        countries.append(v)
    except:
        countries.append('no_country')
    print(k, v)
    
# Getting the only countries in a list

only_countries = countries[1::3]

# Appending the location to the dataframe

was['location'] = only_countries

# Getting the impact, which was a magnitude in the JSON file

magnitude = data_1[2::3]
magnitude = magnitude.reset_index(drop = True)
was['impact'] = magnitude

# Going with the second list

data_2 = tags_adj[1]
latitude = data_2[0::3]
longitude = data_2[1::3]
latitude = latitude.reset_index(drop = True)
longitude = longitude.reset_index(drop = True)
was_1 = pd.DataFrame()
was_1['longitude'] = longitude
was_1['latitude'] = latitude
x = was_1['longitude'].tolist()
y = was_1['latitude'].tolist()
z = list(zip(y, x))
location = rg.search(z)
countries = []
for k, v in [(k, v) for x in location for (k, v,) in x.items()]:
    print(type(v))
    try:
        countries.append(v)
    except:
        countries.append('no_country')
    print(k, v)
    
only_countries = countries[1::3]
was_1['location'] = only_countries
magnitude = data_2[2::3]
magnitude = magnitude.reset_index(drop = True)
was_1['impact'] = magnitude
```

Processing with the data for 'Victims'

```
# Importing the file

df_kasp = pd.read_csv('Kaspersky.csv')

# Working with the file

df_attacks = df_kasp[['Country', 'Estimate_year']]

# Getting the insights of the data

df_attacks.describe()
df_attacks.dtypes

# Creating a new column

df_attacks['Scaled'] = df_attacks['Estimate_year'].copy()

# Importing the library to rescale the column and processing with it

from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()
df_attacks['Scaled_with_minmax'] = scaler_minmax.fit_transform(df_attacks[['Scaled']])

# New dataframe for merging 

df_attacks_merge = df_attacks[['Country', 'Scaled_with_minmax']].copy()
df_attacks_merge = df_attacks_merge.rename(columns={'Country': 'location'})

# Merging two dataframes and processing with it

joint = df_attacks_merge.merge(was, on = 'location',how='right')
joint['Scaled_with_minmax'].fillna(0.0, inplace = True)

# Working with trial dataframe

trial=pd.DataFrame(columns=joint.columns)
for country in countries:
    a = joint[joint['location'] == country]
    a['new_impact'] =  scaler_minmax.fit_transform(a[['impact']])
    trial=trial.append(a,ignore_index=True)
    
trial['test_hack'] = trial['Scaled_with_minmax'] * trial['new_impact']

# Getting the dataframe for export

to_export = trial[['latitude', 'longitude', 'test_hack']]

# Prepare the files for export

nah = to_export.values.tolist()
flat_list = []
for sublist in nah:
    for item in sublist:
        flat_list.append(item)
       
# JSON file

data = [{0: 1990, 1: flat_list}]
nah_nah = pd.DataFrame(data)
nah_nah.to_json('try.json')
```

### Preparing 'Aggressors' Data

```
# Library

import pandas as pd

# Importing the file

df = pd.read_csv('AWS_hacking.csv')

# Getting the insights of the data

df.shape
df.isna().sum()
df.head()

# Working with dataframe

new = df['host'].str.split("-", n = 1, expand = True) 
df['target'] = new[1]

# Dropping the columns

df.drop(['src', 'proto', 'type', 'spt', 'dpt', 'srcstr','Unnamed: 15', 'host'], axis = 1, inplace = True)

# Next steps

new2 = df['datetime'].str.split(" ", n = 1, expand = True) 
df['date'] = new2[0]
df.drop(['datetime'], axis = 1, inplace = True)
df = df.dropna(subset=['cc'])
df.drop(['locale', 'localeabbr', 'postalcode', 'cc'], axis = 1, inplace = True)
df = df.rename(columns = {'country': 'from'})
df = df[['date', 'target', 'from', 'latitude', 'longitude']]
aggression = df['from'].value_counts()

# Importing the library and working with it

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
aggression['scaled_freq'] = scaler.fit_transform(aggression[['freq']])

# Getting the coordinates for the file

imported = pd.read_csv('to_import.csv')
imported.drop(['Unnamed: 0', 'Scaled_with_minmax', 'impact'], axis = 1, inplace = True)

# Processing with it

part = imported[['location', 'latitude', 'longitude', 'new_impact']]
join = part.merge(aggression, on = 'location', how = 'left')
join[['freq', 'scaled_freq']] = join[['freq', 'scaled_freq']].fillna(value=0)
join['damage'] = join['new_impact'] * join['scaled_freq']
join.drop(['location', 'new_impact', 'freq', 'scaled_freq'], axis = 1, inplace = True)

# Preparing the JSON file

nah = join.values.tolist()
flat_list = []
for sublist in nah:
    for item in sublist:
        flat_list.append(item)
        
data = [{0: 1990, 1: flat_list}]
nah_nah = pd.DataFrame(data)
nah_nah.to_json('attacked.json')
```

## Questions

In case you have any questions, you are free to send me an email to ivolgin@protonmail.ch
