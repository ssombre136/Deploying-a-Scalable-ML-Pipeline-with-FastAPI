import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
# TODO: add necessary import
import numpy as np
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
   # TODO: implement the function
    model = RandomForestClassifier()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Return the trained model
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Use the predict method of the RandomForestClassifier to make predictions
    preds = model.predict(X)


    # Return the predictions
    return preds
    
def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    with open(path, 'rb') as f:
        model = pickle.load(f)

    # Return the loaded model
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # Filter the data to include only the rows with the specified categorical feature value
    X_slice = data[data[column_name] == slice_value].drop(columns=[column_name, label])
    y_slice = data[data[column_name] == slice_value][label]

    # Check if all of the categorical features are present in the X_slice DataFrame
    missing_features = set(categorical_features) - set(X_slice.columns)
    if missing_features:
        raise ValueError(f"Missing categorical features: {missing_features}")

    # Extract the categorical features from the filtered data
    X_categorical = X_slice[categorical_features]

    # One-hot encode the categorical features
    X_categorical = encoder.transform(X_categorical)

    # Extract the numerical features from the filtered data
    X_numerical = X_slice.drop(columns=categorical_features)

    # Combine the one-hot encoded categorical features and the numerical features
    X_slice = hstack([X_categorical, X_numerical])

    # Make predictions on the filtered data
    y_pred = model.predict(X_slice)

    # Compute the precision, recall, and F1 score for the filtered data
    p, r, fb, _ = compute_model_metrics(y_slice, y_pred, lb)

    return p, r, fb
