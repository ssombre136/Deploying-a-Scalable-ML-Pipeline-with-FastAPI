import pytest
# TODO: add necessary import
import unittest
import tain, test from train_model

# TODO: implement the first test. Change the function name and input as needed
def test_data_nulls(data)
    """
    test to see if the data has any null values, if it does not it will return that it passed
    """
    assert data.shape == data.dropna().shape, "Dropping null changes shape"



# TODO: implement the second test. Change the function name and input as needed
def test_slice_averave():
    """
    test to see if the mean (or average) of each categorical slice is between 1.5 and 2.5
    """
    for cat_features in data["categorical_features"].unique():
        avg_value = data[data["categorical_features"] == cat_features]["numeric_feat"].mean()
            assert(
                2.5 > avg_value > 1.5
                ), f"For {cat_features}, average of {avg_values} not between 2.5 and 3.5"


# TODO: implement the third test. Change the function name and input as needed
def test_train_model():
    """
    this will test to make sure that train_model returns the expected retult
    """
    train_model_retult = train_model()
    
    expected_train_model_result = train_model()
    assert train_model_retult == expected_train_model_result
