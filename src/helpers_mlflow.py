"""Helper file for modelling scripts"""

############################### LIBRARIES #####################################

# MLFLOW
import mlflow
from mlflow.tracking import MlflowClient

# Database libraries
import pymysql

# Command line libraries
import subprocess
from threading import Timer


############################### MYSQL DATABASE SETUP #####################################

def create_database_storage(dbServerName, dbPort, dbName, dbUser, dbPassword):

    """
    Create a database in MYSQL using SQLAlchemy engine with the pymysql package

    Args:
        dbServerName (str): The Server Name
        dbPort (str): The Port Number
        dbName (str): The database name used for storing the MLflow model tracking output
        dbUser (str): The Username of the database
        dbPassword (str): The Password of the database

    """

    # Create a connection object
    charSet             = "utf8mb4"
    cusrorType          = pymysql.cursors.DictCursor
    connectionObject    = pymysql.connect(host=dbServerName, port= dbPort, user=dbUser, password=dbPassword,
                                            charset=charSet,cursorclass=cusrorType)

    try:
        # Create a cursor object
        cursorObject        = connectionObject.cursor()
        # Create the database
        sqlQuery            = f"CREATE DATABASE IF NOT EXISTS {dbName};"   
        # Execute the sqlQueryc
        cursorObject.execute(sqlQuery)
            
    except Exception as e:
        print("Exeception occured:{}".format(e))

    finally:
        connectionObject.close()
        print(f"Database created at MYSQL: {dbName} on {dbUser}:{dbPassword}@{dbServerName}/{dbPort}")


def show_databases(dbServerName, dbUser, dbPassword):

    """
    Checker function to see if the database was created

    Args:
        dbServerName (str): The Server Name
        dbUser (str): The Username of the database
        dbPassword (str): The Password of the database

    Returns:
        All the database within the connected SQL server
    """

    # Create a connection object
    charSet         = "utf8mb4"
    cusrorType      = pymysql.cursors.DictCursor
    connectionObject   = pymysql.connect(host=dbServerName, user=dbUser, password=dbPassword,
                                        charset=charSet,cursorclass=cusrorType)
    
    print("Databases available")
    try:
        # Create a cursor object
        cursorObject        = connectionObject.cursor()
        # SQL query string
        sqlQuery            = "SHOW DATABASES;"
        # Execute the sqlQuery
        cursorObject.execute(sqlQuery)

        # Fetch all the rows
        rows = cursorObject.fetchall()
        for row in rows:
            print(row)
            
    except Exception as e:
        print("Exeception occured:{}".format(e))

    finally:
        connectionObject.close()


############################### MLFLOW SETUP #####################################

def create_mlflow_cmd(storage_filepath, dbName, dbUser, dbPassword, dbServerName, dbPort):
    
    """
    Creates the command line to initialise the Mlflow setup

    Args:
        storage_filepath (str): The Mlflow experiment name configuration
        dbName (str): The database name used for storing the MLflow model tracking output
        dbUser (str): The Username of the database
        dbPassword (str): The Password of the database
        dbServerName (str): The Server Name, default host is 127.0.0.1
        dbPort (str): The Port Number, default is 3306

    Returns:
        mlflow_cmd (str): The generated command line given the specified parameters
    """

    if dbServerName == None:
        dbServerName = '127.0.0.1'
    if dbPort == None:
        dbPort = 3306

    mlflow_cmd = f'''mlflow ui \
                    --backend-store-uri {storage_filepath} \
                    --registry-store-uri mysql+pymysql://{dbUser}:{dbPassword}@localhost:{dbPort}/{dbName} \
                    --host {dbServerName} --port 5000 \
                    --serve-artifacts'''

    return mlflow_cmd


def run_cmd(cmd, timeout_sec):

    """
    Run a command line in python for a certain timeout period

    Args:
        cmd (str): The Shell command line in string. eg. "echo Hello World"
        timeout_sec (int): The period to run the command line before the process is killed
    """

    # Trigger the subprocess that runs the command line
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Initialise the timer with the time out
    timer = Timer(int(timeout_sec), proc.terminate())
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()

def setup_mlflow(EXPERIMENT_NAME, ARTIFACT_REPO):

    """
    Initialises the MLflow setup for UI and model registry access.
    The model registry (MLflow tracking) is connected to a SQL database as per Scenario 2 in the documentation.
    [https://mlflow.org/docs/1.30.1/tracking.html#how-runs-and-artifacts-are-recorded]

    Args:
        EXPERIMENT_NAME (str): The Mlflow experiment name configuration
        ARTIFACT_REPO (str): The filepath for Mlflow artifact storage

    Returns:
        experiment_id (str): The Mlflow experiment ID created
        client (obj): The Mlflow client referencing the UI tracking URI. Default is "http://127.0.0.1:5000/".
    """

    TRACKING_URI = "http://127.0.0.1:5000/"

    client = MlflowClient(TRACKING_URI) # Initialize client
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLFLOW UI is at: {mlflow.get_tracking_uri()}")
    
    # Get the experiment id if it already exists and if not create it
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_REPO)
    except Exception as err:
        print(err)
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    return experiment_id, client

############################### REGISTERING MODELS #####################################

def mlflow_get_run_from_registered_model(client, registry_model_name, stage="Staging"):
    """
    For a given registered model, return the MLflow ModelVersion object based on its name and stage
    This contains all metadata needed, such as params logged, Runid etc

    Args:
        registry_model_name (str): Name of MLflow Registry Model
        stage (str): Stage for this model. One of "Staging" or "Production"

    Returns:
        run (str): Run which holds params logged, Runid objects, etc. specific to that model's specified stage
        run_id (str): Run id specific to that model's specified stage
    """

    # client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_models = client.search_registered_models(filter_string=filter_string)
    
    if not registered_models:
        return None, None

    for model_version in registered_models[0].latest_versions:
        if model_version.current_stage == stage:
            run_id = model_version.run_id
            run = mlflow.get_run(run_id)
            return run, run_id

    return None, None


def evaluate_model_performance_for_registry(client, registry_model_name, pos_metrics_list, pos_metrics_thresh_diff_list, neg_metrics_list, neg_metrics_thresh_diff_list):
    """
    Evaluate the performance of the existing model and recommend whether to transition the staging model to production.

    Args:
        registry_model_name (str): Model name for MLflow registration.
        pos_metrics_list (list): List of positive metrics (eg. RMSE).
        pos_metrics_thresh_diff_list (list): Threshold differences for positive metrics (good if prod-stage < threshold).
        neg_metrics_list (list): List of negative metrics (eg. MAE).
        neg_metrics_thresh_diff_list (list): Threshold differences for negative metrics (good if stage-prod < threshold).

    Returns:
        model_registry_ind (str): Indicates whether to register the model in production ('yes' or 'no').
    """   

    # Get the registered production model
    prod_existing_model, prod_model_run_id = mlflow_get_run_from_registered_model(client, registry_model_name, stage="Production")

    # Get the registered staging model
    current_staging_run, stage_model_run_id = mlflow_get_run_from_registered_model(client, registry_model_name, stage="Staging") 
      
    # Initialize metric lists
    pos_metric_list, neg_metric_list = [], []
    
    for pos_metric, threshold in zip(pos_metrics_list, pos_metrics_thresh_diff_list):
        stage_pos_metric = current_staging_run.data.metrics[str(pos_metric)]
        prod_pos_metric = prod_existing_model.data.metrics[str(pos_metric)]
        
        print(f":\nStage_{pos_metric}:", stage_pos_metric, f"\nProd_{pos_metric}:", prod_pos_metric,
              "\nProd-Stage Diff:", prod_pos_metric - stage_pos_metric, "\nProd-Stage Threshold Diff:", threshold)
        
        pos_metric_list.append(stage_pos_metric > prod_pos_metric or (prod_pos_metric - stage_pos_metric) < threshold)
        
    for neg_metric, threshold in zip(neg_metrics_list, neg_metrics_thresh_diff_list):
        stage_neg_metric = current_staging_run.data.metrics[str(neg_metric)]
        prod_neg_metric = prod_existing_model.data.metrics[str(neg_metric)]
        
        print(f"\nStage_{neg_metric}:", stage_neg_metric, f"\nProd_{neg_metric}:", prod_neg_metric,
              "\nStage-Prod Diff:", stage_neg_metric - prod_neg_metric, "\nStage-Prod Threshold Diff:", threshold)
        
        neg_metric_list.append(stage_neg_metric < prod_neg_metric or (stage_neg_metric - prod_neg_metric) < threshold)
    
    print(pos_metric_list, neg_metric_list)

    if all(pos_metric_list) and all(neg_metric_list):
        model_registry_ind = 'yes'
        print("\nThe current staging provides better metrics values, and the metric differences are within acceptable thresholds."
              "\nThis staging model will be transitioned to production.")
    else:
        model_registry_ind = 'no'
        print("\nNOTE: The current staging run performs worse compared to the existing model, so the model would not be registered")      

    return model_registry_ind


def mlflow_register_new_model_in_mlflow(model_run_info, registry_model_name):

    """ 
    Register 1st time MLflow Model to its "None" stage by its name and run_info
    
    Args:
        model_run_info (String) : MLFLOW Experiment run detail to register teh model
        registry_model_name (String) : Model name  to register in mlflow

    Returns:
        model_run_version: Registered model's version number
    """ 

    model_run_id = model_run_info.info.run_id
    model_run_version = mlflow.register_model(model_uri=f"runs:/{model_run_id}/model", name=registry_model_name)
    
    return model_run_version


def mlflow_transition_model(client, model_version, stage):
    """
    This function Transition a model to a specified stage in MLflow Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    Args:
        model_version (String) : MLFLOW model version which needs to be transited
        stage (String)         : New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"

    Returns:
        stage_model_version (String) : MLFLOW transited stage model version (Which remains same as before transit version)
    """
    
    stage_model_version = client.transition_model_version_stage(name=model_version.name,
                                                                version=model_version.version,
                                                                stage=stage,
                                                                archive_existing_versions=True )

    return stage_model_version


def mlflow_existing_model_compare_and_registry(client, model_run_info, registry_model_name,
                                               pos_metrics_list=[], pos_metrics_thresh_diff_list=[],
                                               neg_metrics_list=[], neg_metrics_thresh_diff_list=[], 
                                               model_type='None'):
    """ 
    Compare the MLflow model to the current model and, if the new model is better, register it in the production model registry. 
    For models with model_type 'None', always register them for production.

    Args:
        client: the MLFlow client object
        model_run_info (str): MLFLOW Experiment's run objects.
        registry_model_name (str): Model name to register in MLflow.
        pos_metrics_list (list): Positive metrics list (e.g., RMSE).
        pos_metrics_thresh_diff_list (list): Threshold differences for positive metrics (good if prod-stage < threshold).
        neg_metrics_list (list): Negative metrics list (e.g., MAE).
        neg_metrics_thresh_diff_list (list): Threshold differences for negative metrics (good if stage-prod < threshold).
        model_type (str): Model type ('Classification', 'Regression', or 'None').

    Returns:
        model_production_version (str): Registered model's production version number.
    """ 

    prod_existing_model, model_run_id = mlflow_get_run_from_registered_model(client, registry_model_name, stage="Production")

    # If no registered model in production, assign new model as production model
    if prod_existing_model is None:
        print("No registered model in MLFLOW production.")
        print("First time training model in production, it will be automatically assigned as production model.")
        
        model_run_version = mlflow_register_new_model_in_mlflow(model_run_info, registry_model_name)
        model_staging_version = mlflow_transition_model(client, model_run_version, stage="Staging")
        model_production_version = mlflow_transition_model(client, model_staging_version, stage="Production")
        print("New model has been registered and transitioned to production.")

    else: # Evaluate existing model

        print("Existing registered production model is available. Run_id is: {} \n".format(model_run_id))

        # Register the new model
        model_run_version = mlflow_register_new_model_in_mlflow(model_run_info, registry_model_name)
        model_staging_version = mlflow_transition_model(client, model_run_version, stage="Staging")
        print("New model has been registered and transitioned to staging.")

        # Perform model comparison
        if model_type == 'None':
            model_registry_ind = 'yes'
            print("Model type is 'None', so directly registering in production (No comparison).")
        else:
            model_registry_ind = evaluate_model_performance_for_registry(client, registry_model_name, pos_metrics_list, pos_metrics_thresh_diff_list, neg_metrics_list, neg_metrics_thresh_diff_list)

        if model_registry_ind == 'yes':
            model_production_version = mlflow_transition_model(client, model_staging_version, stage="Production")
            print("New model has been compared and transitioned to production.")  
        else:
            print("The new model does not perform better than the existing production model, so it will not be registered in production.")

    return model_production_version


############################### END OF SCRIPT #####################################