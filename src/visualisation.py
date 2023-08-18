from matplotlib import pyplot as plt

def multiply_values(acc_list: list) -> list:
    output_list = []
    for x in acc_list:
        output_list.append(x*100)
    return output_list

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    acc = multiply_values(acc)
    val_acc = multiply_values(val_acc)


    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = list(range(1,len(loss)+1))

    fig = plt.figure(figsize=(12, len(loss)+4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xticks(epochs_range)
    plt.xlabel('Epochs')
    plt.ylim(0, 100)
    plt.ylabel("Accuracy %")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xticks(epochs_range)
    plt.xlabel('Epochs')
    plt.ylim(0, max(val_loss+loss)*1.1)
    plt.ylabel("Loss")

    return fig
