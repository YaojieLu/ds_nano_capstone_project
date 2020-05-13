# Q1: who spent most
# Q2: who viewed most
# Q3: who completed which offer the most
# Q4: who spent during which offer the most
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split#, GridSearchCV

# read in files
df = pd.read_csv('data/df.csv')