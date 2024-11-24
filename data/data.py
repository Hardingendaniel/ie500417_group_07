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
    with open(file_path, 'r', encoding='utf-8') as file: # In order to read Ñ properly in the markdown format, added encoding to UTF-8!
        return file.read()

def load_global_temp():
    global_temp = pd.read_csv('data/monthly-average-surface-temperatures-by-year.csv')
    return global_temp

def load_ghg_release():
    ghg_release = pd.read_csv('data/ghg-emissions-by-gas.csv')
    return ghg_release

# Load all the datasets
data = load_data()  # Load raw OWID CO2 data
ghg_data = load_total_ghg_data()  # Load Total GHG Emissions dataset
global_temp_data = load_global_temp() # Load the global temperature datasetss
ghg_release =load_ghg_release() # Load the ghg release ratio dataset.
