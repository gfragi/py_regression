import pandas as pd
from sklearn.model_selection import train_test_split


# Assign the url where the dataset reside 
delaney_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv'

# Read the dataser and assign it to a variable
delaney_df = pd.read_csv(delaney_url)

# Assign the url where the dataset reside 
delaney_descriptors_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'

# Read the calculated descriptors
delaney_descriptors_df = pd.read_csv(delaney_descriptors_url)

print(delaney_descriptors_df)

# Drop logS which is the y variable 
x = delaney_descriptors_df.drop('logS', axis=1)

# Assign to y variable the logS column
y = delaney_descriptors_df.logS

# In evaluate the model performance split e the dataset into 2 partitions (80% - 20% ration)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

