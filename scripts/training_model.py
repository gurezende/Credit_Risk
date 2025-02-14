#%%
# Importing libraries
# Data wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Statistics
import scipy.stats as scs

# Dataset
from ucimlrepo import fetch_ucirepo 

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# Evaluation
from sklearn.metrics import mean_squared_error, root_mean_squared_error, classification_report, r2_score

# MLFlow
import mlflow

# %%
# Connecting with MLFlow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set Experiment ID (Get from MLFlow Experiment Tab)
mlflow.set_experiment(experiment_id=993400020501733055)

# %%

# fetch dataset 
# statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# # data (as pandas dataframes)  X and y
# X = statlog_german_credit_data.data.features 
# y = statlog_german_credit_data.data.targets 

# # Make it a data frame for easier manipulation
# df = pd.concat([X,y], axis=1)

# # Rename columns according to the documentation
# df.columns = ['is_customer', 'duration_mths', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employed', 'installment_rate_pct',
#               'sex_status', 'guarantors', 'same_resid_since', 'property', 'age', 'other_installment_plans', 'housing', 'n_credits_this_bank',
#               'job', 'dependents', 'phone', 'foreign_worker', 'target']

# # dataset information 
# print(f'Data shape (rows,cols): {df.shape}')
# print('---')
# print(f'There are --{df.isna().sum().sum()}-- missing values in the dataset.')
# print('---')
# df.info()

#%%
df2 = pd.read_csv('../.data/credits.csv')


# %%

# X and y for regression
Xr = df2.drop(['credit_amt', 'sex_status', 'target', 'guarantors', 'phone', 
               'foreign_worker', 'other_installment_plans' ], axis=1)
yr = pd.DataFrame(scs.boxcox(df2['credit_amt'])[0], columns=['credit_amt'])
_, lbd = scs.boxcox(df2['credit_amt'])

 
 
# Split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(Xr, yr, 
                                                    test_size=0.1, 
                                                    random_state=42)

# %%

### Preprocessing pipeline

# numerical features
num_features = X_train.select_dtypes(include='number').columns.tolist()
num_steps = [('scaler', StandardScaler())]

# categorical features
cat_features = X_train.select_dtypes(include='object').columns.tolist()
cat_steps = [('encoder', OneHotEncoder(drop='first'))]

# Pipeline
preprocess_pipe = ColumnTransformer([
    ('num', Pipeline(num_steps), num_features),
    ('cat', Pipeline(cat_steps), cat_features)
    ])


# %%
### Final Pipeline

modeling = Pipeline([
    ('preprocess', preprocess_pipe),
    # ('rfe', RFE(estimator=LinearRegression(), n_features_to_select=n)),
    ('model', LinearRegression())
    ])

# Set the description as a tag
description = "df2 Less Variables2 Target Box-Cox model"

# Start MLFlow Run:
with mlflow.start_run(description=description):
    # Fit
    modeling.fit(X_train, y_train)
    # Ask MLFlow to log the basic information about the model
    mlflow.sklearn.autolog()
    # Model evaluation
    y_hat = modeling.predict(X_train)
    y_pred = modeling.predict(X_test)
    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, y_hat)
    rmse_ = root_mean_squared_error(y_test, y_pred)
    # Log additional metrics (custom for my project)
    mlflow.log_metrics({'test_rmse':rmse_,
                        'test_r2': r2_score(y_test, y_pred)})
    
# %%

### Using XGBoost Regressor

lgb_reg = Pipeline([
    ('preprocess', preprocess_pipe),
    ('model', lgb.LGBMRegressor(
        objective='regression',  # For regression tasks
        n_estimators=300,      # Number of boosting rounds
        learning_rate=0.015,     # Learning rate
        num_leaves=35,         # Maximum number of leaves for base learners
        random_state=42,       # Random seed for reproducibility
        n_jobs=-1,
        ) )
    ])

# Set the description as a tag
description = "df2 LGBM model 300 | 35 leaves | 0.015 lr"

# Start MLFlow Run:
with mlflow.start_run(description=description):
    # Fit
    lgb_reg.fit(X_train, y_train)
    # Ask MLFlow to log the basic information about the model
    mlflow.sklearn.autolog()
    # Model evaluation
    y_hat = lgb_reg.predict(X_train)
    y_pred = lgb_reg.predict(X_test)


    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, y_hat)
    rmse_ = root_mean_squared_error(y_test, y_pred)
    
    # Log additional metrics (custom for my project)
    mlflow.log_metrics({'test_rmse':rmse_,
                        'test_r2': r2_score(y_test, y_pred)})
# %%
from scipy.special import inv_boxcox
pred_df = inv_boxcox(y_test, lbd)
pred_df['pred'] = inv_boxcox(y_pred, lbd)
pred_df['errors'] = pred_df['credit_amt'] - pred_df['pred']
pred_df.sample(10)
print(f"Mean Error: {pred_df['errors'].mean()}")

# %%
import seaborn as sns
sns.histplot(pred_df['errors'], kde=True)
# %%
scs.shapiro(pred_df['errors'])

# %%
scs.probplot(pred_df['errors'], dist='norm', plot=plt);


# %%
