# Social Network Ad based Predictions

## Setting up the environment


```python
import pandas as pd
df = pd.read_csv('../Social-Network-Ads/Social_Network_Ads.csv')
# check for number of null entries
print('Data is packed' if not df.isnull().values.any() else 'Data contains missing values')
df.head()
```

    Data is packed





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15624510</td>
      <td>Male</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15810944</td>
      <td>Male</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15668575</td>
      <td>Female</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15603246</td>
      <td>Female</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15804002</td>
      <td>Male</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizing the data

### Heatmap


```python
import seaborn as sb
import matplotlib.pyplot as plt

sb.heatmap(df.corr(), annot = True, cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.show()
```


![png](ANN%20Logistic%20Regressor_files/output_5_0.png)


### Interpretable graph


```python
from matplotlib.colors import ListedColormap

# Age vs salary. Men are blue points, Females are Red points and No Buyers are black points
salaries = df['EstimatedSalary']
ages = df['Age']
genders = df['Gender']
buy_stats = df['Purchased']
colors = [1 if genders[i] == 'Male' and buy_stats[i] == 1 else 2 if genders[i] == 'Female' and buy_stats[i] == 1 else 0 for i in range(len(df['User ID'].tolist()))]
cmap = ListedColormap(['black','blue','red'])
classes = ['Not Bought', 'Male', 'Female']
plt.clf()

fig = plt.figure()
fig.suptitle("Impact of salary and age")
scatter = plt.scatter(salaries, ages, c = colors, cmap = cmap)
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.xlabel('Salary-->')
plt.ylabel('Age-->')
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](ANN%20Logistic%20Regressor_files/output_7_1.png)


## Preprocessing the data

### Data Cleaning and Scaling


```python
# User ID is irrelevant
df = df.drop(['User ID'], axis = 1)

# We can change Male and Female to 0 or 1
df['Gender'] = df['Gender'].apply(lambda gender: 0 if gender == 'Male' else 1)

# Normalize Age and EstimatedSalary based on mean and range
# value = (val - mean)/(max - min)

# Salary Normalization
import math
salaries = df['EstimatedSalary']
mean_salary = min(salaries) + max(salaries) / len(salaries)
salary_range = max(salaries) - min(salaries)
norm_salaries = [(salary - mean_salary) / salary_range for salary in salaries]

# Age normalization
ages = df['Age']
mean_age = min(ages) + max(ages) / len(ages)
age_range = max(ages) - min(ages)
norm_ages = [(age - mean_age) / age_range for age in ages]

# Assign to df
df['Age'] = norm_ages
df['EstimatedSalary'] = norm_salaries
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.020238</td>
      <td>0.026852</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.401190</td>
      <td>0.034259</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.186905</td>
      <td>0.204630</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.210714</td>
      <td>0.308333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.020238</td>
      <td>0.449074</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Train and Test split


```python
# Use 60% data as training data and 20% as validation data and 20% as test data
y = df.pop('Purchased').to_numpy()
X = df.to_numpy()

# Split
from sklearn.model_selection import train_test_split

# Train+Validation and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Train and Validation split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25)
```


```python
print(f'''Training size: {len(X_train)}
Validation size: {len(X_valid)}
Test size: {len(X_test)}''')
```

    Training size: 240
    Validation size: 80
    Test size: 80


# Creating Keras Model


```python
from tensorflow import keras

#ANN
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(X_train.shape[1], )))
model.add(keras.layers.Dense(int(X_train.shape[1]*2), input_dim=X_train.shape[1], activation='relu'))
model.add(keras.layers.Dense(X_train.shape[1]*2, activation = 'relu'))
model.add(keras.layers.Dense(X_train.shape[1]*2, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 3)                 0         
    _________________________________________________________________
    dense (Dense)                (None, 6)                 24        
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 42        
    _________________________________________________________________
    dense_2 (Dense)              (None, 6)                 42        
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 7         
    =================================================================
    Total params: 115
    Trainable params: 115
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
#Set epochs
epochs = 100

#Clear output
import IPython
class ClearTrainingOutput(keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output()
    
#Fit the model
trained_model = model.fit(X_train, y_train, epochs = epochs, validation_data = (X_valid, y_valid), callbacks = [ClearTrainingOutput()])
```


```python
print('Max Accuracy Reached: {0:0.2f} %'.format((max(trained_model.history['accuracy']) * 100)) )
# Print the learning curve
plt.plot(range(1, epochs+1), trained_model.history['loss'] , 'r', range(1, epochs+1), trained_model.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
```

    Max Accuracy Reached: 84.17 %





    <matplotlib.legend.Legend at 0x7f320b9c9ca0>




![png](ANN%20Logistic%20Regressor_files/output_17_2.png)


# Making Predictions


```python
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
predictions = [0 if prob < 0.5 else 1 for prob in predictions]
print("Test Set Accuracy: {0:0.2f} %".format(accuracy_score(y_test, predictions) * 100))
```

    Test Set Accuracy: 85.00 %


# Creating auto-tuned Keras model

## Defining model with variable hyperparameters


```python
from tensorflow import keras
import kerastuner as kt
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(X_train.shape[1], )))
    for i in range(hp.Int('layers', 1, 10)):
        model.add(keras.layers.Dense(
            units=hp.Int('units_' + str(i), X_train.shape[1], X_train.shape[1]*5, step = X_train.shape[1]),
            activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
        
    model.compile(optimizer = 'adam',
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])
    return model

tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = 20,
                     factor = X_train.shape[1],
                     directory = 'auto_tuned_model',
                     project_name = 'soc_net_ad_pred')

tuner.search(X_train, y_train, epochs = 20, validation_data = (X_valid, y_valid), callbacks = [ClearTrainingOutput()])
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
```

    Trial 30 Complete [00h 00m 02s]
    val_accuracy: 0.8125
    
    Best val_accuracy So Far: 0.8125
    Total elapsed time: 00h 00m 51s
    INFO:tensorflow:Oracle triggered exit



```python
## Create model from tuned hypermodel
model = tuner.hypermodel.build(best_hps)
print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 3)                 0         
    _________________________________________________________________
    dense (Dense)                (None, 12)                48        
    _________________________________________________________________
    dense_1 (Dense)              (None, 15)                195       
    _________________________________________________________________
    dense_2 (Dense)              (None, 15)                240       
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 16        
    =================================================================
    Total params: 499
    Trainable params: 499
    Non-trainable params: 0
    _________________________________________________________________
    None


## Train the model with best hyper parameters


```python
epochs = 100
trained_model = model.fit(X_train, y_train, epochs = epochs, validation_data = (X_valid, y_valid), callbacks = [ClearTrainingOutput()])
```


```python
print('Max Accuracy Reached: {0:0.2f} %'.format((max(trained_model.history['accuracy']) * 100)) )
# Print the learning curve
plt.plot(range(1, epochs+1), trained_model.history['loss'] , 'r', range(1, epochs+1), trained_model.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
```

    Max Accuracy Reached: 87.92 %





    <matplotlib.legend.Legend at 0x7f31c4137790>




![png](ANN%20Logistic%20Regressor_files/output_26_2.png)


## Making predictions with auto-tuned model


```python
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
predictions = [0 if prob < 0.5 else 1 for prob in predictions]
print("Test Set Accuracy: {0:0.2f} %".format(accuracy_score(y_test, predictions) * 100))
```

    Test Set Accuracy: 91.25 %

