![Ironhack logo](https://i.imgur.com/1QgrNNw.png)

# Cybercrime

This is my final project at Ironhack, which helped me get to the Final Hackshow on the 13th of December.

## Final Presentation

For the final presentation please check the following Google Slides: 
https://docs.google.com/presentation/d/113M_4-i2eDk3lb1LXbMqv33i7r_5MPepVMrfasY1NUQ/edit?usp=sharing

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
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
```


## World Globe

Working with this type of Visualization was a bit new for me, that is why I've adjusted only the data source without changing a lot of the source code. The data file was in JSON format, so I've imported it into Jupyter Notebook and have done the adjustments there.

The Globe visualization consists of two different parts: victims and aggressors

### Data Sources:

* [CYBERTHREAT REAL-TIME MAP](https://cybermap.kaspersky.com) - Victims
* [Kaggle](https://www.kaggle.com/casimian2000/aws-honeypot-attack-data) - AWS Honeypot Attack Data

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
