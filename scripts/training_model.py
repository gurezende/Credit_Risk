#%%
# Importing libraries
# Data wrangling
import pandas as pd
import numpy as np

# Statistics
import scipy.stats as scs

# Dataset
from ucimlrepo import fetch_ucirepo 

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Evaluation
from sklearn.metrics import mean_squared_error, root_mean_squared_error, classification_report
# %%

# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes)  X and y
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 

# Make it a data frame for easier manipulation
df = pd.concat([X,y], axis=1)

# Rename columns according to the documentation
df.columns = ['is_customer', 'duration_mths', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employed', 'installment_rate_pct',
              'sex_status', 'guarantors', 'same_resid_since', 'property', 'age', 'other_installment_plans', 'housing', 'n_credits_this_bank',
              'job', 'dependents', 'phone', 'foreign_worker', 'target']

# dataset information 
print(f'Data shape (rows,cols): {df.shape}')
print('---')
print(f'There are --{df.isna().sum().sum()}-- missing values in the dataset.')
print('---')
df.info()


# %%

# X and y for regression
Xr = df_encoded.drop(['credit_amt', 'target'], axis=1)
yr = pd.DataFrame(scs.boxcox(df_encoded['credit_amt'])[0], columns=['credit_amt'])
 
 
# Split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(Xr, yr, 
                                                    test_size=0.1, 
                                                    random_state=42)

# %%


