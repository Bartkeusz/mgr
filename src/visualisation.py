import matplotlib as plt

def show_chart(plot_data):
    plot_data.plot(x='label', y='count', kind='bar', color='steelblue', legend=False)
    plt.xlabel('Zjawisko atmosferyczne')
    plt.ylabel('Liczba wystąpień')
    plt.show()

def show_classification_report(y_test, x_test, model, labels):
    _y_pred = model.predict(x_test)
    _y_pred = np.argmax(_y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(classification_report(y_test, _y_pred, target_names=labels))

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = list(range(1,len(loss)+1))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()