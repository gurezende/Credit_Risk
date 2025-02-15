import pandas as pd
import numpy as np

# Function to determine the customer balance
def get_balance(balance):
    """Function to determine the customer balance range."""

    try:
        balance = float(balance)  # Attempt conversion to float
    except ValueError:
        return "no checking account"  # Handle cases where conversion fails

    if balance < 0:
        return "<0"
    elif 0 <= balance <= 200000:  # More Pythonic range check
        return "0 - 200k"
    elif balance > 200000:
        return ">=200k"  # Removed extra quotes
    else:  # This 'else' is unlikely to be reached now, but good practice
        return "no checking account"  # Consider what this really means
        
