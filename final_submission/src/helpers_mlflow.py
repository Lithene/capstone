"""Helper file for modelling scripts"""

############################### LIBRARIES #####################################

# MLFLOW
import mlflow
from mlflow.tracking import MlflowClient

# Modelling libraries
# Importing Gensim
import gensim
from gensim import corpora
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import linear_model
from sklearn.metrics import f1_score

############################### MODELLING #####################################
### MLFLOW
def setup_mlflow():
    EXPERIMENT_NAME = "ai_critic"
    ARTIFACT_REPO = './aicritic_mlflow'
    client = MlflowClient() # Initialize client
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    
    # Get the experiment id if it already exists and if not create it
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_REPO)
    except Exception as err:
        print(err)
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    return experiment_id


def mlflow_get_run_from_registered_model(registry_model_name, stage="Staging"):
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

    client = MlflowClient()
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


def evaluate_model_performance_for_registry(registry_model_name, pos_metrics_list, pos_metrics_thresh_diff_list, neg_metrics_list, neg_metrics_thresh_diff_list):
    """
    Evaluate the performance of the existing model and recommend whether to transition the staging model to production.

    Args:
     registry_model_name (str): Model name for MLflow registration.
     pos_metrics_list (list): List of positive metrics (e.g., RMSE).
     pos_metrics_thresh_diff_list (list): Threshold differences for positive metrics (good if prod-stage < threshold).
     neg_metrics_list (list): List of negative metrics (e.g., MAE).
     neg_metrics_thresh_diff_list (list): Threshold differences for negative metrics (good if stage-prod < threshold).

    Returns:
     model_registry_ind (str): Indicates whether to register the model in production ('yes' or 'no').
    """   

    # Get the registered production model
    prod_existing_model, prod_model_run_id = mlflow_get_run_from_registered_model(registry_model_name, stage="Production")

    # Get the registered staging model
    current_staging_run, stage_model_run_id = mlflow_get_run_from_registered_model(registry_model_name, stage="Staging") 
      
    # Initialize metric lists
    pos_metric_list, neg_metric_list = [], []
    
    for pos_metric, threshold in zip(pos_metrics_list, pos_metrics_thresh_diff_list):
        stage_pos_metric = current_staging_run.data.metrics[str(pos_metric)]
        prod_pos_metric = prod_existing_model.data.metrics[str(pos_metric)]
        
        print("\n", pos_metric, ":\nStage:", stage_pos_metric, "\nProd:", prod_pos_metric,
              "\nProd-Stage Diff:", prod_pos_metric - stage_pos_metric, "\nProd-Stage Threshold Diff:", threshold)
        
        pos_metric_list.append(stage_pos_metric > prod_pos_metric or (prod_pos_metric - stage_pos_metric) < threshold)
        
    for neg_metric, threshold in zip(neg_metrics_list, neg_metrics_thresh_diff_list):
        stage_neg_metric = current_staging_run.data.metrics[str(neg_metric)]
        prod_neg_metric = prod_existing_model.data.metrics[str(neg_metric)]
        
        print("\n", neg_metric, ":\nStage:", stage_neg_metric, "\nProd:", prod_neg_metric,
              "\nStage-Prod Diff:", stage_neg_metric - prod_neg_metric, "\nStage-Prod Threshold Diff:", threshold)
        
        neg_metric_list.append(stage_neg_metric < prod_neg_metric or (stage_neg_metric - prod_neg_metric) < threshold)
    
    print(pos_metric_list, neg_metric_list)

    if all(pos_metric_list) and all(neg_metric_list):
        model_registry_ind = 'yes'
        print("\nThe current staging provides better metrics values, and the metric differences are within acceptable thresholds."
              "\nYou may proceed with transitioning the staging model to production now.")
    else:
        model_registry_ind = 'no'
        print("\nNOTE: The current staging run performs worse compared to the existing model, so we won't register this model")      

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


def mlflow_transition_model(model_version, stage):
    """
    This function Transition a model to a specified stage in MLflow Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    Args:
     model_version (String) : MLFLOW model version which needs to be transited
     stage (String)         : New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"

    Returns:
     stage_model_version (String) : MLFLOW transited stage model version (Which remains same as before transit version)
    """

    client = MlflowClient()
    
    stage_model_version = client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )

    return stage_model_version



def mlflow_existing_model_compare_and_registry(model_run_info, registry_model_name,
                                               pos_metrics_list=[], pos_metrics_thresh_diff_list=[],
                                               neg_metrics_list=[], neg_metrics_thresh_diff_list=[], 
                                               model_type='None'):
    """ 
    Compare the MLflow model to the current model and, if the new model is better, register it in the production model registry. 
    For models with model_type 'None', always register them for production.

    Args:
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

    prod_existing_model, model_run_id = mlflow_get_run_from_registered_model(registry_model_name, stage="Production")

    # If no registered model in production, treat it as a new model
    if prod_existing_model is None:
        print("No registered model in MLFLOW production.")
        print("It's the first time training this model in production, so it satisfies the basic test and will be moved into staging and production.")
        
        model_run_version = mlflow_register_new_model_in_mlflow(model_run_info, registry_model_name)
        model_staging_version = mlflow_transition_model(model_run_version, stage="Staging")
        model_production_version = mlflow_transition_model(model_staging_version, stage="Production")
        print("New model has been registered and transitioned to production.")

    else: # Existing Model

        print("Existing registered production model is available. Read that and perform model comparison. Run_id is: {} \n".format(model_run_id))

        # Register the new model
        model_run_version = mlflow_register_new_model_in_mlflow(model_run_info, registry_model_name)
        model_staging_version = mlflow_transition_model(model_run_version, stage="Staging")
        print("New model has been registered and transitioned to staging.")

        # Perform model comparison
        if model_type == 'None':
            model_registry_ind = 'yes'
            print("Model type is 'None', so directly registering in production (No comparison).")
        else:
            model_registry_ind = mlflow_compare_models_perfs(registry_model_name, pos_metrics_list, pos_metrics_thresh_diff_list, neg_metrics_list, neg_metrics_thresh_diff_list)

        if model_registry_ind == 'yes':
            model_production_version = mlflow_transition_model(model_staging_version, stage="Production")
            print("New model has been compared and transitioned to production.")  
        else:
            print("The new model does not perform better than the existing production model, so it will not be registered in production.")

    return model_production_version


############################### END OF SCRIPT #####################################