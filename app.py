# Imports
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
from scipy.special import inv_boxcox
from scripts.utils import get_balance

#-------------------------------------------------------

# Load credit data for charts
df = pd.read_csv('./.data/credits.csv')

#-------------------------------------------------------
# Making connection to the MLFlow server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Start a MLFlow Client to get the latest model version
client = mlflow.client.MlflowClient()

# Getting latest version
version = 4 # version 4 is the Linear Regression with less variables

# Import the latest model version
model = mlflow.sklearn.load_model(f"models:/Credit_Regression/{version}")

#-------------------------------------------------------

# Create an empty DataFrame to store the inputs
to_predict = pd.DataFrame(columns=model.feature_names_in_)

#The model needs these columns as inputs:
# 'is_customer', 'duration_mths', 'credit_hist', 'purpose', 'savings', 'employed',
# 'installment_rate_pct', 'guarantors','same_resid_since', 'property', 'age', 
# 'other_installment_plans', 'housing', 'n_credits_this_bank', 'job', 'dependents',
# 'phone','foreign_worker', 'sex', 'status'

#-------------------------------------------------------
# APPLICATION
#-------------------------------------------------------

def main():

    # Create the input form
    st.title('Pre-Approved Credit Amount')
    st.subheader('Fill in the form to get your pre-approved credit amount')

    #-------------------------------------------------------

    # Creating two columns for the input form
    col1, col2, col3 = st.columns(3)

    with col1:
        
        # is_customer
        is_customer = st.selectbox('Are you a customer?', ['Yes', 'No'])
        # if selection == Yes then enable input box for balance
        if is_customer == 'Yes':
            balance = st.number_input(label = 'Balance',
                                    min_value=-1e100, 
                                    value=0.0,
                                    step=1.0)
        else:
            balance = 'no checking account'

        customer = get_balance(balance)

        #---
        # duration_mths
        if is_customer == 'Yes':
            duration_mths = st.number_input(label = 'Customer for how many months?',
                                            min_value=1., 
                                            value=1.,
                                            step=1.)
        else:
            duration_mths = 0

        #---
        # credit_hist
        credit_hist = st.selectbox(label='Credit History',                          
                                options=['critical account/  other credits existing (not at this bank)',
                                    'exist paid till now', 
                                    'delay in paying off in the past',
                                    'no credits taken/ all paid',
                                    'this bank all paid'],
                                    index=None,
                                    placeholder="Choose an option")


        #---
        # purpose
        purpose = st.selectbox(label='Purpose of Loan',                          
                            options=['radio/television',
                                        'education', 
                                        'furniture/equipment',
                                        'car (new)',
                                        'car (used)',
                                        'business',
                                        'domestic appliances',
                                        'repairs',
                                        'others',
                                        'retraining'],
                                index=None,
                                placeholder="Choose an option")

        #---
        #savings
        savings = st.selectbox(label='Savings',                          
                            options=['unknown/ no savings account',
                                        '< 100k',
                                        '500k - 1M',
                                        '>= 1M',
                                        '100k - 500k'],
                                index=None,
                                placeholder="Choose an option")

    #-------------------------------------------------------

    with col2:

        #---
        #employed
        employed = st.selectbox(label='Employment Time',                          
                            options=['>= 7 years',
                                        '1 - 4 years  ',
                                        '4 - 7 years',
                                        'unemployed',
                                        '< 1 year'],
                                index=None,
                                placeholder="Choose an option")

        #installment_rate_pct
        installment_rate = st.number_input(label = 'Installment Rate %',
                                min_value=0.,
                                value=0.0,
                                step=0.1)
        
        #---
        # guarantors (removed)
        # guarantors = st.selectbox(label='Guarantors',
        #                           options=['none', 'co-applicant', 'guarantor'],
        #                           index=None,
        #                           placeholder="Choose an option")

        #---
        # same_resid_since
        same_resid_since = st.number_input(label='Same Residence for How Long',               
                                        min_value=0.,
                                        value=1.,
                                        step=1.)
                                        
        #---
        # property
        property = st.selectbox(label='Property',                               
                                options=['real estate',
                                        'if not A121 : building society savings agreement/ life insurance',
                                        'unknown / no property',
                                        'if not A121/A122 : car or other, not in attribute 6'],
                                index=None,
                                placeholder="Choose an option")


        # age
        age = st.number_input(label = 'Age',
                            min_value=18.,
                            value=18.,
                            step=1.)

        #---
        # other_installment_plans (removed)
        # other_installment_plans = st.selectbox(label='Other installment plans',
        #                                     options=['bank', 'stores', 'none'],
        #                                     index=None,
        #                                     placeholder="Choose an option")

        #---
        # housing
        housing = st.selectbox(label='Housing',                               
                            options=['rent', 'own', 'for free'],
                            index=None,
                            placeholder="Choose an option")

    #-------------------------------------------------------

    with col3:
        #---
        # n_credits_this_bank
        n_credits_this_bank = st.number_input(label = 'Number of Credits',
                                            min_value=0.,
                                            value=0.,
                                            step=1.)

        #---
        # job
        job = st.selectbox(label='Job',                               
                        options=['skilled employee / official',
                                'unskilled - resident',
                                'management/ self-employed/ highly qualified employee/ officer',
                                'unemployed/ unskilled  - non-resident'],
                        index=None,
                        placeholder="Choose an option")

        #---
        # dependents
        dependents = st.number_input(label = 'Dependents',
                                    min_value=0.,
                                    value=0.,
                                    step=1.)

        #---
        # phone (removed)
        # phone = st.selectbox(label='Phone',                               
        #                     options=['no', 'yes'],
        #                     index=None,
        #                     placeholder="Choose an option")

        #---
        # foreign_worker (removed)
        # foreign_worker = st.selectbox(label='Foreign Worker',                               
        #                             options=['no', 'yes'],
        #                             index=None,
        #                             placeholder="Choose an option")

        #---
        # sex
        sex = st.selectbox(label='Sex',                               
                        options=['male', 'female'],
                        index=None,
                        placeholder="Choose an option")

        #---
        # status
        status = st.selectbox(label='Status',                               
                            options=['single',
                                    'divorced/separated/married',
                                    'divorced/separated',
                                    'married/widowed'],
                            index=None,
                            placeholder="Choose an option")

    #-------------------------------------------------------

    # Submit button
    if st.button('Submit'):

        # Add to Dataframe
        new_obs = pd.DataFrame({
            'is_customer': customer,
            'duration_mths': duration_mths,
            'credit_hist': credit_hist,
            'purpose': purpose,
            'savings': savings,
            'employed': employed,
            'installment_rate_pct': installment_rate,
            # 'guarantors': guarantors,
            'same_resid_since': same_resid_since,
            'property': property,
            'age': age ,
            # 'other_installment_plans': other_installment_plans,
            'housing': housing,
            'n_credits_this_bank': n_credits_this_bank,
            'job': job,
            'dependents': dependents,
            # 'phone': phone,
            # 'foreign_worker': foreign_worker,
            'sex': sex,
            'status': status
            }, index=[0])

        to_predict = pd.concat([to_predict, new_obs])


        # Model prediction
        y_pred = model.predict(to_predict)

        # Revert Box-Cox
        lbd = -0.06401037626058795

        # Create a DataFrame with the predictions
        pred = inv_boxcox(y_pred, lbd)

        # Prediction formatted
        f_pred = round(pred[0][0],2)

        # Add space
        st.write('---')

        #-------------------------------------------------------
        # CREDIT REPORT
        #-------------------------------------------------------

        # Estimated Credit
        st.title('Pre-Approved Credit Amount')
        st.subheader(f'Predicted Credit Amount: ${f_pred:,}')

        st.write('')
        st.write('')

        # graphics
        # Filter the dataset within this range of the estimated credit 10% lower and 10% higher
        df_filtered = df.query('credit_amt >= @f_pred*0.9 and credit_amt <= @f_pred * 1.1')

        # Gender Distribution
        gender_dist = (df_filtered
                    .sex
                    .value_counts(normalize=True, ascending=False)
                    .to_frame()
                    .reset_index()
                    .head(1) )
        
        # Main Purpose
        main_purpose = (df_filtered
                    .purpose
                    .value_counts(ascending=False)
                    .to_frame()
                    .reset_index()
                    .head(1)
                    .purpose )

        # Create Report Columns
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        
        # column 1: Main purpose for credit request
        with rcol1:
            st.text('MAIN PURPOSE')
            st.subheader( f':orange[{str.upper(main_purpose[0])}]' )

        # column 2 : How many credit lines in the range 10% lower and 10% higher
        with rcol2:
            st.text('CREDIT LINES OF\nSIMILAR AMOUNT')
            st.title( f':orange[{df_filtered.shape[0]}]' )

        #column 3 : % Risk in this range
        with rcol3:
            st.text('RISK IN\nTHIS RANGE')
            st.title(f":orange[{(
                df_filtered['target']
                .value_counts(normalize=True)[0]
                .round(2)*100)
                .astype('str')+'%'}]" 
                )
            
        # column 4 : Gender Distribution
        with rcol4:
            st.text(f'GENDER DISTRIBUTION\nMAJORITY {str.upper(gender_dist.sex[0])}')
            st.title(f':orange[{gender_dist.proportion[0].round(2)*100}%]')

        st.button('Reset')

    else:
        st.subheader('Predicted Credit Amount: $0.00')

if __name__ == '__main__':
    main()