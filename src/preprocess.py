import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def split_data_into_train_and_test(x, y, number_of_classes: int,  path_to_config: str = "read_data_config.json"):
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config["test_size"], random_state=config["random_state"])
    y_train = to_categorical(y_train, number_of_classes)
    y_test = to_categorical(y_test, number_of_classes)
    return x_train, x_test, y_train, y_test