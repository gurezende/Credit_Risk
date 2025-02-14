#%%
import mlflow
import pandas as pd
from scipy.special import inv_boxcox
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


#%%
# Making connection to the MLFlow server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Start a MLFlow Client to get the latest model version
client = mlflow.client.MlflowClient()

# Getting latest version
version = 2#client.get_model_version_by_alias("Credit_Regression", alias="1").version

# Import the latest model version
model = mlflow.sklearn.load_model(f"models:/Credit_Regression/{version}")

#%%

# Prediction Dataset (Test)
new_X = pd.read_csv('../.data/X_test.csv')
new_y = pd.read_csv('../.data/y_test.csv')

#%%
# Model prediction
y_pred = model.predict(new_X)

# Revert Box-Cox
lbd = -0.06401037626058795

# Create a DataFrame with the predictions
pred_df = inv_boxcox(new_y, lbd)
pred_df['pred'] = inv_boxcox(y_pred, lbd)
pred_df['errors'] = pred_df['credit_amt'] - pred_df['pred']
pred_df.sample(10)

# %%
print(f"RMSE: {root_mean_squared_error(pred_df['credit_amt'],
                                       pred_df['pred'])}")

print(f"MAE: {mean_absolute_error(pred_df['credit_amt'],
                                       pred_df['pred'])}")



# %%
