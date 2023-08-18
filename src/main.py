from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import numpy as np
from mlflow_extend import mlflow
from sklearn.metrics import roc_auc_score, f1_score
import json
from urllib3.exceptions import NewConnectionError

from read_data import read_data, prepare_dataset
from preprocess import split_data_into_train_and_test
import models
from visualisation import plot_history


def run(path_to_config: str = "read_data_config.json") -> None:
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    
    labels, number_of_classes = read_data()
    x, y = prepare_dataset(labels)
    x_train, x_test, y_train, y_test = split_data_into_train_and_test(x, y, number_of_classes)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test1")

    experiment = mlflow.get_experiment_by_name("test1")

    # with mlflow.start_run(run_name=f"test_run_2", experiment_id=experiment.experiment_id):
    #     mlflow.tensorflow.autolog()
    #     model = models.UseResNet50model(number_of_classes, config['batch_size'], config['image_size'], x[0].shape, config['patience'], config['optimizer'])
    #     model.build_model()
    #     history = model.train_model(x_train, y_train, x_test, y_test)


    with mlflow.start_run(run_name=f"test_run_3", experiment_id=experiment.experiment_id) as run:
        mlflow.tensorflow.autolog()
        model = models.UseVGGmodel(number_of_classes, config['batch_size'], config['image_size'], x[0].shape, config['patience'], config['optimizer'])
        model.build_model()
        history, y_pred = model.train_model(x_train, y_train, x_test, y_test)
        roc_auc = roc_auc_score(y_test, y_pred, average="weighted", multi_class='ovr')
        report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=labels, output_dict=True)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_dict(report, "classification_report.json")
        fig = plot_history(history)
        mlflow.log_figure(fig, "history.png")
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    run()