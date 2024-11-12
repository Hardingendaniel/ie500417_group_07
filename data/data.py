# data.py
import pandas as pd

# Load and prepare the data once
def load_data():
    data = pd.read_csv('data/owid-co2-data.csv')
    data = data.dropna(subset=['co2'])
    return data

# Load data and store it in a global variable
data = load_data()

