from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import logging
import multiprocessing
import os

file_dir = os.path.dirname(__file__)

logging.basicConfig(filename=os.path.join(file_dir, '../logs/ml_model.log'),
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


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

    parameters = {
        'n_estimators': [10, 15, 20],
        'max_depth': [2, 5, 10],
        'min_samples_split': [20, 50, 100],
        'max_features': ['auto', 'sqrt'],
    }

    n_jobs = multiprocessing.cpu_count() - 1
    logging.info(
        f"Finding optimum values of hyperparameters on {n_jobs} cores.")

    rfc_model = GridSearchCV(RandomForestClassifier(random_state=0),
                             param_grid=parameters,
                             cv=3,
                             n_jobs=n_jobs,
                             verbose=2,
                             )

    rfc_model.fit(X_train, y_train)
    logging.info(f"BEST PARAMS: {rfc_model.best_params_}")

    return rfc_model


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
    predictions = model.predict(X)
    return predictions


def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix on predictions.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, preds)
    return cm


def compute_slice_metrics(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed

    Inputs
    ------
    df:
        test dataframe pre-processed with features as column used for slices
    feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized

    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(
        index=slice_options,
        columns=[
            'feature',
            'n_samples',
            'precision',
            'recall',
            'fbeta'])
    for option in slice_options:
        slice_mask = df[feature] == option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)

        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature value', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] = colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df
