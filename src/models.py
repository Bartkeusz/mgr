from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class UseResNet50model:
    def __init__(self, number_of_classes, batch_size, img_size, dim, patience, optimizer):
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.dim = dim
        self.patience = patience
        self.optimizer = optimizer

    def build_model(self):
        self.model = ResNet50(
        include_top=False,
        weights="imagenet",
        classifier_activation="relu",
        input_shape = self.dim,)

        self.model.trainable = False


        flatten_layer = layers.Flatten()
        prediction_layer = layers.Dense(self.number_of_classes, activation='softmax')

        self.model = models.Sequential([
            self.model,
            # Conv2D(32, (3,3), input_shape=(self.img_size, self.number_of_classes), activation='relu', padding='same'),
            # Conv2D(32, (3,3), activation='relu', padding='same'),
            # MaxPooling2D(2,2),
            # Conv2D(64, (3,3), activation='relu', padding='same'),
            # Conv2D(64, (3,3), activation='relu', padding='same'),
            # MaxPooling2D(2,2),
            # Conv2D(128, (3, 3), activation = 'relu', padding='same'),
            flatten_layer,
            prediction_layer
            ])
        

        
        self.model.summary()

        self.model.compile(
          optimizer=self.optimizer,
          loss=['categorical_crossentropy','mse'],
          metrics=['accuracy', 'CategoricalAccuracy']
          )

    def train_model(self, x_train, y_train, x_test, y_test):
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=self.patience, restore_best_weights=True)
        self.history = self.model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=self.batch_size, callbacks=[es])

        model_accuracy = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, return_dict=True)

        return self.history, model_accuracy
    

class UseVGGmodel:
    def __init__(self, number_of_classes, batch_size, img_size, dim, patience, optimizer):
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.dim = dim
        self.patience = patience
        self.optimizer = optimizer

    def build_model(self):
        self.model =  VGG16(
        include_top=False,
        weights="imagenet",
        classifier_activation="relu",
        input_shape = self.dim,)

        self.model.trainable = False

        flatten_layer = layers.Flatten()
        prediction_layer = layers.Dense(self.number_of_classes, activation='softmax')

        self.model = models.Sequential([
            self.model,
            flatten_layer,
            prediction_layer
            ])
        
        self.model.summary()

        self.model.compile(
          optimizer=self.optimizer,
          loss='categorical_crossentropy',
          metrics=['accuracy']
          )

    def train_model(self, x_train, y_train, x_test, y_test):
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=self.patience, restore_best_weights=True)
        self.history = self.model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=self.batch_size, callbacks=[es])

        model_accuracy = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, return_dict=True)

        return self.history, model_accuracy