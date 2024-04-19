import pytest
# TODO: add necessary import
import os
import unittest
from train_model import train
from train_model import test
from train_model import cat_features, model, preds
from ml.model import train_model, performance_on_categorical_slice
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

project_path = "/Users/rowanzinn/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
csv_path = os.path.join(project_path, "data", "census.csv")

# TODO: implement the first test. Change the function name and input as needed
def test_data_nulls():
    """
    test to see if the data has any null values, if it does not it will return that it passed
    """
    assert test.shape == test.dropna().shape, "Dropping null changes shape"



# TODO: implement the second test. Change the function name and input as needed
def test_inference_returns_expected_type():

    assert isinstance(preds, np.ndarray), "inference did not return a numpy array"



# TODO: implement the third test. Change the function name and input as needed
def test_train_model():
    """
    this will test to make sure that train_model returns the expected retult
    """
    assert isinstance(model, RandomForestClassifier), "train_model did not return a RandomForestClassifier object"
    
#trying something to fix a action error
def test_model(csv_path):
    """
    test to make sure that the model can be trained and tested using the provided CSV file
    """
    data = pd.read_csv(csv_path)
    # Add your test code here
test_model(csv_path)
    
