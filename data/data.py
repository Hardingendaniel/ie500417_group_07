# data.py
import pandas as pd

# Load and prepare the data once
import pandas as pd

# Load and prepare the OWID CO2 data
def load_data():
    """Load and clean the OWID CO2 dataset."""
    data = pd.read_csv('data/owid-co2-data.csv')
    data = data.dropna(subset=['co2'])
    return data

# Load the Total GHG Emissions dataset
def load_total_ghg_data():
    """Load the Total GHG Emissions dataset."""
    ghg_data = pd.read_csv('data/total-ghg-emissions.csv')
    return ghg_data

# Load and read markdown files
def load_markdown(file_path):
    """Read and return the content of a Markdown file."""
    with open(file_path, 'r') as file:
        return file.read()

def load_global_temp():
    global_temp = pd.read_csv('data/monthly-average-surface-temperatures-by-year.csv')
    return global_temp

# Load all the datasets
data = load_data()  # Load raw OWID CO2 data
ghg_data = load_total_ghg_data()  # Load Total GHG Emissions dataset
global_temp_data = load_global_temp() # Load the global temperature dataset
