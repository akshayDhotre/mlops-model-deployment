# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Prediction task is to determine whether a person makes over 50K a year.

Model use `RandomForestClassifier` from `sklearn.model.RandomForestClassifier` for classification tasks.
Hyperparameters tuning was realized using GridSearchCV.

Model is saved as a pickle file in the model folder. All training steps and metrics are logged in the file "logs/ml_model.log".

## Intended Use
This model can be used to predict the salary level of an individual based on cnesus data having few attributes. The usage is meant for students, academics or research purpose.


## Training Data
The Census Income Dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) as a csv file.
The original data set has 32.561 rows and 15 columns composed of the target label "salary", 8 categorical features and 6 numerical features.
Details on each of the features ae available at the UCI link above.
Target label "salary" has two classes ('<=50K', '>50K') and shows class imbalance with a ratio of circa 75% / 25%.
A simple data cleansing was performed on the original dataset to remove leading and trailing whitespaces. See data_cleaning.ipynb notebook for data exploration and cleansing step.

A 80-20 split was used to break this dataset into a train and test set. Stratification on target label "salary" was applied.
To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the target.


## Evaluation Data
20% of the dataset was set aside for model evaluation.
Transformation was applied on the categorical features and the target label respectively using the One Hot Encoder and label binarizer fitted on the train set.

## Metrics
The classification performance is evaluated using precision, recall and fbeta metrics.
 - precision: 0.794
 - recall: 0.542
 - fbeta: 0.644

The confusion matrix is also calculated.
 - [[4320  211][ 688  814]]


## Ethical Considerations
The dataset should not be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories.

## Caveats and Recommendations
Extraction was done from the 1994 Census database. The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems. 