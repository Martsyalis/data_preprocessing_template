import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset 
dataset = pd.read_csv('./Data.csv')
x = dataset.iloc[:, :-1].values # grab values in all rows and all but the last columns
y = dataset.iloc[:, -1].values # grab values in all rows for the last column

# Split set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 1) # split 80/20 and seed random with 1