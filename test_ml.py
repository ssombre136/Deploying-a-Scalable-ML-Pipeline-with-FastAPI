import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ml.model import train_model, compute_model_metrics, inference

# TODO: implement the first test. Change the function name and input as needed
def test_model_returns_randomforestclassifier():
    """
    Using dummy variables tests if the training model returns a RandomForestClassifier
    """
    # Your code here
    row_train = np.random.rand(20, 5)
    col_train = np.random.randint(1, 2, size=20)
    model = train_model(row_train, col_train)
    assert isinstance(model, RandomForestClassifier)
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_model_inference():
    """
    Using dummy variables tests if model can perform inference and return correct shape.
    """
    row_train = np.random.rand(20, 5)
    col_train = np.random.randint(1, 2, size=20)
    row_test = np.random.rand(5, 5)
    model = train_model(row_train, col_train)
    predictions = inference(model, row_test)
    assert predictions.shape == (5,)
    assert isinstance(predictions, np.ndarray)
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_model_metrics():
    """
    Using dummy variables verifies that the model's accuracy metric is as expected.
    """
    # Your code here
    row_train = np.random.rand(20, 5)
    col_train = np.random.randint(1, 2, size=20)
    row_test = np.random.rand(20, 5)
    col_test = np.random.randint(1, 2, size=20)
    model = train_model(row_train, col_train)
    prediction = model.predict(row_test)
    accuracy = compute_model_metrics(col_test, prediction)
    assert np.all(np.isclose(accuracy, accuracy_score(col_test, prediction)))
    
    pass
