from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np
import mlflow
from urllib3.exceptions import NewConnectionError

from read_data import read_data, prepare_dataset
from preprocess import split_data_into_train_and_test
import models
from visualisation import plot_history



def run() -> dict:
    labels, number_of_classes = read_data()
    x, y = prepare_dataset(labels)
    x_train, x_test, y_train, y_test = split_data_into_train_and_test(x, y, number_of_classes)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("pierwsze uruchomienie")
    experiment = mlflow.get_experiment_by_name("pierwsze uruchomienie")
    with mlflow.start_run(run_name=f"test_run", experiment_id=experiment.experiment_id):
        mlflow.tensorflow.autolog()
        
        model = models.UseResNet50model(number_of_classes, 16, [224, 224], x[0].shape, 3)
        model.build_model()
        history = model.train_model(x_train, y_train, x_test, y_test)
        print(history)
        plot_history(history)


if __name__ == "__main__":
    run()