from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from visualisation import show_classification_report

class UseResNet50model:
    def __init__(self, number_of_classes, batch_size, img_size, dim, patience):
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.dim = dim
        self.patience = patience

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
            flatten_layer,
            prediction_layer
            ])
        
        self.model.summary()

        self.model.compile(
          optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy']
          )

    def train_model(self, x_train, y_train, x_test, y_test):
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=self.patience, restore_best_weights=True)
        self.history = self.model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=self.batch_size, callbacks=[es])

        model_accuracy = self.model.evaluate(x_test, y_test, batch_size=self.batch_size, return_dict=True)
        print(f"ResNet50 model accuracy: {model_accuracy}")
        print(show_classification_report(y_test, x_test, self.model))

        return self.history
    