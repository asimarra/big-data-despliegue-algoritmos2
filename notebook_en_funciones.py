import argparse
import subprocess
import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def argumentos():
    print("aaa")
    parser = argparse.ArgumentParser(description='__main__ de la aplicación con argumentos de entrada.')
    parser.add_argument('--nombre_job', type=str, help='Valor para el parámetro nombre_documento.', required=False, default="pruebas-desde-script-default")
    parser.add_argument('--n_estimators_list', nargs='+', type=int, help='List of n_estimators values.', required=True)
    return parser.parse_args()

def load_dataset():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']
    return df

def data_treatment(df):
    # Split data into train and test sets
    train, test = train_test_split(df, test_size=0.2)
    test_target = test['target']
    test[['target']].to_csv('test-target.csv', index=False)
    del test['target']
    test.to_csv('test.csv', index=False)

    features = [x for x in list(train.columns) if x != 'target']
    x_raw = train[features]
    y_raw = train['target']
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                        test_size=.20,
                                                        random_state=123,
                                                        stratify=y_raw)
    return x_train, x_test, y_train, y_test

def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, n_estimators):

    time.sleep(5)
    mlflow.set_experiment(nombre_job)
    for i in n_estimators:
        with mlflow.start_run() as run:
            clf = RandomForestClassifier(n_estimators=i,
                                          min_samples_leaf=2,
                                          class_weight='balanced',
                                          random_state=123)

            preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

            model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('RandomForestClassifier', clf)])
            model.fit(x_train, y_train)
            accuracy_train = model.score(x_train, y_train)
            model.score(x_test, y_test)

            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_param('n_estimators', i)
            mlflow.sklearn.log_model(model, 'clf-model')
    print("Se ha acabado el entrenamiento del modelo correctamente")