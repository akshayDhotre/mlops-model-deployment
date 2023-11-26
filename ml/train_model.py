"""
Script to train machine learning model.
"""

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pickle
import logging
import pandas as pd
from data import process_data
from model import train_model, inference, compute_model_metrics
from model import compute_confusion_matrix, compute_slice_metrics


# Add code to load in the data.
file_dir = os.path.dirname(__file__)
datapath = pd.read_csv(os.path.join(file_dir, './data/census.csv'))

# model save path artifacts
modelpath = './model'
filename = ['rfc_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Initialize logging
logging.basicConfig(filename=os.path.join(file_dir, './logs/ml_model.log'),
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')


# Function to remove files given the path
def remove_if_exists(filename):
    """
    Delete a file if it exists.
    input:
        filename: str - path to the file to be removed
    output:
        None
    """
    if os.path.exists(filename):
        os.remove(filename)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
# Set train flag to False to use the encoding from the train set
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

# Check if saved model exist and load them
if os.path.isfile(os.path.join(file_dir, modelpath,filename[0])):
        model = pickle.load(open(os.path.join(file_dir, modelpath,filename[0]), 'rb'))
        encoder = pickle.load(open(os.path.join(file_dir, modelpath,filename[1]), 'rb'))
        lb = pickle.load(open(os.path.join(file_dir, modelpath,filename[2]), 'rb'))

# Train and save a model if does not exist
else:
    rfc_model = train_model(X_train, y_train)

    model_path = os.path.join(file_dir, modelpath, filename[0])
    pickle.dump(rfc_model, open(model_path, 'wb'))

    encoder_path = os.path.join(file_dir, modelpath, filename[1])
    pickle.dump(encoder, open(encoder_path, 'wb'))

    lb_path = os.path.join(file_dir, modelpath, filename[2])
    pickle.dump(lb, open(lb_path, 'wb'))

# Evaluate the model

preds = inference(rfc_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")

logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))

logging.info(f"Confusion matrix:\n{cm}")

# Compute performance on slices for categorical features
slice_savepath = "./slice_output.txt"
remove_if_exists(slice_savepath)

# iterate through the categorical features and save results to log and txt file
for feature in cat_features:
    performance_df = compute_slice_metrics(test, feature, y_test, preds)
    performance_df.to_csv(slice_savepath,  mode='a', index=False)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)