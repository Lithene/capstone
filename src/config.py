"""Config File for filepaths and other hardcoded variables that extends to all scripts"""

############################### LIBRARIES #####################################
import os
import pandas as pd
from datetime import date

################################# DATES ########################################

today = date.today().strftime("%y%m%d")

############################### DATA SCRAPPING #####################################

# Instagram Login Config
insta_username = ''
insta_pwd = ''

# Set Main Directory
main_directory = ""
# Raw data storage
raw_data_path = os.path.join(main_directory, "raw_data")

# Raw Dataset Output filepaths
insta_user_datapath     = os.path.join(raw_data_path, f"instagraphi_bio_{today}.csv")
insta_media_datapath    = os.path.join(raw_data_path, f"instagraphi_media_{today}.csv")
fb_media_datapath       = os.path.join(raw_data_path, f"facebook_{today}.csv")
breachlist_datapath     = os.path.join(raw_data_path, f"breach_list_{today}.pkl")

# Feature dataset
feature_data            = os.path.join(raw_data_path, f"full_features_{today}.csv")

############################### MLFLOW MYSQL #####################################

# MYSQL connection object configurations
dbServerName        = "127.0.0.1"
dbPort              = 3306
dbUser              = "root"
dbPassword          = ""
dbName              = "mlflow_tracking_database"

# Configuration for MLFlow to load
db_connection_str   = f"mysql+pymysql://{dbUser}:{dbPassword}@localhost/{dbName}"
storage_filepath    = "file:/./aicritic_mlflow"
exp_name            = "ai_critic"
timeout             = 20

############################### MODELLING #####################################
# Temporary file storage for artifacts logging and model registering
output_path             = os.path.join(main_directory, "working_dir")
# Saving dataset from train / test split to replicate results
data_artifact_path      = os.path.join(output_path, "data")
# Saving encoders used for one-hot encoding / target enocding
encoder_artifact_path   = os.path.join(output_path, "encoder")
# Saving model artifacts for Machine Learning model
ml_artifact_path        = os.path.join(output_path, "mlflow")
# Saving model artifacts for Neural Network model
nn_artifact_path        = os.path.join(output_path, "nn_model")
# Saving word tokenizer
tokenizer_artifact_path = os.path.join(output_path, "tokenizer")
# Load glove file
glove_file              = "inputs\\glove.6B.100d.txt"
# Saving predictions and model reasonings
exai_artifact_path      = os.path.join(output_path, "predictions")


############################### INFERENCE #####################################

inference_output = os.path.join(main_directory, "inference_output")

############################### CREATE FILEPATHS #####################################

def create_path(filepath: str):
    """
    Creates the specified filepath if directory does not exists.
    Args:
        filepath (Str): filepath to create
    """
    
    if not os.path.exists(filepath):
        print(f"Creating filepath {filepath}")
        os.makedirs(filepath)
    else:
        print("Filepath already exists")

############################### DATA LOAD/EXPORT #####################################

def get_latest_csv(filepath: str, filename_filter: str):
    """
    Retrieve the current csv file as per the filter string from the specified filepath.
    Args:
        filepath (Str): The filepath to be imported from. Eg. "/path/to/filename.csv"
        filename_filter (Str): The file name filter eg. "full_features"
    Returns:
        dataset (pandas.DataFrame): The retrieved dataset

    """

    # Get list of all files
    all_files = [os.path.join(filepath, x) for x in os.listdir(filepath) if x.startswith(filename_filter) and x.endswith(".csv")]
    # Get latest file based on load date
    curr_filepath = max(all_files, key = os.path.getctime)
    dataset = pd.read_csv(curr_filepath, index_col= None)
    print(f"Dataset from {curr_filepath}")

    return dataset

def export_file_csv(dataframe, export_path: str, mode: str='x'):
    """
    Export the dataframe as a csv into specified filepath.
    Args:
        dataframe (pandas.DataFrame): The dataframe to be exported
        export_path	(Str): The filepath to be exported into. Eg. "/path/to/filename.csv"
        mode (Str): The mode of csv write
                    - 'x': default, no overwriting; exclusive file creation
                    - 'w+': overwrite mode
    """

    today = date.today().strftime("%y%m%d")
    
    try:
        dataframe.to_csv(export_path, index=False, mode=mode)
        print(f"Exported df {dataframe.shape} as {export_path}")
            
    except Exception as e:
        print(f"Could not export file as {e}")
