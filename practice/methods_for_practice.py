import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline

def ask_params():
    parser = argparse.ArgumentParser(description='__main__ of the practice')
    parser.add_argument('--job_name', type=str, help='job name', required=False, default="test-from-script")
    parser.add_argument('--n_estimators_list', nargs='+', type=int, help='List of n_estimators values.', required=True)
    parser.add_argument('--max_depth', type=int, help='max_depth for the model', required=False, default=20)
    parser.add_argument('--max_features', type=int, help='max_features for the model', required=False, default=10000)
    parser.add_argument('--class_weight', type=str, help='class_weight for the model', required=False, default=None)
    return parser.parse_args()

def load_clean_dataset():
    file_path = 'cleaned_reviews.csv'
    df = pd.read_csv(file_path)

    # Convert column 'overall' to binary variable (positive if >=4, negative if <=2)
    df['sentiment'] = df['overall'].apply(lambda x: 1 if x >= 4 else 0 if x <= 2 else np.nan)
    df = df.dropna()  # clean null values

    print("loaded clean dataset")
    return df

def data_treatment(df):
    if 'clean_review' not in df.columns:
        raise ValueError("The column 'clean_review' is not in the DataFrame.")

    if df['clean_review'].isnull().sum() > 0:
        raise ValueError("The column 'clean_review' contains null values.")

    x_train, x_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])

    print("data_treatment successfully")

    return x_train, x_test, y_train, y_test


def mlflow_tracking(job_name, x_train_vec, x_test_vec, y_train, y_test, args_values):
    print("data_treatment start " + job_name)
    time.sleep(5)
    mlflow.set_experiment(job_name)
    for i in args_values.n_estimators_list:
        with mlflow.start_run() as run:
            rf_model = RandomForestClassifier(
                n_estimators=i, 
                random_state=42,
                max_depth=args_values.max_depth, 
                max_features=args_values.max_features, 
                class_weight=args_values.class_weight,
                n_jobs=-1
            )
            rf_model.fit(x_train_vec, y_train)
            y_predicted_train = rf_model.predict(x_train_vec)
            y_predicted_test = rf_model.predict(x_test_vec)

            accuracy_train = accuracy_score(y_train, y_predicted_train)
            accuracy_test = accuracy_score(y_test, y_predicted_test)
            precision = precision_score(y_test, y_predicted_test)
            recall = recall_score(y_test, y_predicted_test)
            f1 = f1_score(y_test, y_predicted_test)
            roc_auc = roc_auc_score(y_test, y_predicted_test)
            log_loss = log_loss(y_test, y_predicted_test)

            # log params
            mlflow.log_param('random_state', 42)
            mlflow.log_param('n_estimators', i)
            mlflow.log_param('max_depth', args_values.max_depth)
            mlflow.log_param('max_features', args_values.max_features)
            mlflow.log_param('class_weight', args_values.class_weight)
            mlflow.log_param('model_type', 'RandomForestClassifier')

            # log metrics
            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_metric('accuracy_test', accuracy_test)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('roc_auc', roc_auc) # evaluar qué tan bien el modelo distingue entre clases.
            mlflow.log_metric('log_loss', logloss)

    print("Training completed")


def mlflow_tracking_with_pipeline(job_name, x_train, x_test, y_train, y_test, args_values):
    print("data_treatment start " + job_name)
    time.sleep(5)
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(job_name)
    for i in args_values.n_estimators_list:
        print("n_estimator: ", i)
        with mlflow.start_run() as run:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(2, 3), sublinear_tf=True, max_df=0.9, min_df=5)),
                ('svd', TruncatedSVD(n_components=300)),
                ('rf', RandomForestClassifier(n_estimators=i, random_state=42, max_depth=args_values.max_depth, max_features=args_values.max_features, n_jobs=-1))
            ])
            print("pipeline created")
            pipeline.fit(x_train, y_train)
            print("finished pipeline")
            y_predicted_train = pipeline.predict(x_train)
            print("predicted train finished")
            y_predicted_test = pipeline.predict(x_test)
            print("predicted test finished")

            accuracy_train = accuracy_score(y_train, y_predicted_train)
            accuracy_test = accuracy_score(y_test, y_predicted_test)

            mlflow.log_param('random_state', 42)
            mlflow.log_param('n_estimators', i)
            mlflow.log_param('max_depth', args_values.max_depth)
            mlflow.log_param('max_features', args_values.max_features)
            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_metric('accuracy_test', accuracy_test)
            mlflow.sklearn.log_model(pipeline, 'rf-model')
            
    print("Training with pipeline completed")