"""Helper file for plotting"""

############################### LIBRARIES #####################################
import numpy as np
import pandas as pd
import datetime
import time

# Plotting libraries
import matplotlib.pyplot as plt

############################### PERFORMANCE #####################################

def get_decile_score(predictions, ground_truth_col, predict_col, predict_proba_col):

  decile_df = pd.DataFrame()
  decile_df['num_posts'] = predictions.groupby(by=[predict_proba_col]).count()[predict_col]
  decile_df['num_incompliant'] = predictions.groupby(by=[predict_proba_col]).sum()[ground_truth_col]
  decile_df['num_incompliant_undetected'] = abs(predictions.groupby(by=[predict_proba_col]).sum()[predict_col] - predictions.groupby(by=[predict_proba_col]).sum()[ground_truth_col])
  decile_df['perc_undetected'] = round(decile_df['num_incompliant_undetected'] / decile_df['num_posts'] * 100, 2)

  return decile_df.sort_index(ascending= False)

############################### FEATURE IMPORTANCE #####################################

def plot_feature_importance(Coeff):
  fig = px.bar(Coeff,  x='Coefficients', y='Feature', orientation='h', title= "Feature Importance for Logistic Regression")
  fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

  fig.show()

def boxplot_prob_dist(prob_dist, save_dir=None):

  import plotly.express as px
  import os
  import kaleido

  df = px.data.tips()
  fig = px.box(prob_dist, y="probability of incompliancy (%)", title="Correct Predictions on Test Set")
  fig.update_layout(
      autosize=False,
      width=500,
      height=500,)
  
  if save_dir != None:
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig.write_image(save_dir + "boxplot.png")

  fig.show()

############################### LSTM #####################################

def evaluate_lstm(lstm_model, X_test, y_test):
  score = lstm_model.evaluate(X_test, y_test, verbose=1)
  print("Test Accuracy:", score[1])
  print("Test Accuracy in %:", score[1] *100)

  return score[1]

def plot_lstm_performance(lstm_model_history):
  # Model Performance Charts
  import matplotlib.pyplot as plt
  
  plt.plot(lstm_model_history.history['acc'])
  plt.plot(lstm_model_history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train','test'], loc='upper left')
  plt.show()
  plt.plot(lstm_model_history.history['loss'])
  plt.plot(lstm_model_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train','test'], loc='upper left')
  plt.show()


