'''
Splits data into train and test sets, import model and train
'''
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from mobilenetv2_base import mobilev2_base


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 90
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=str, default=1e-4,
                        help='Adam learning rate')
    parser.add_argument('--batch', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum number of epochs to train')
    parser.add_argument('--model_number', type=str, default='1',
                        help='Name of model')
    args = parser.parse_args()

    INIT_LR, BATCH_SIZE, EPOCHS = float(args.lr), args.batch, args.epochs

    # Load processed NumPy objects for train test split
    print('Loading dataset...')
    X_train, y_train = np.load('train/XTrain.npy'), np.load('train/yTrain.npy')
    X_test, y_test = np.load('test/XTest.npy'), np.load('test/yTest.npy')

    # Generate batches of tensor image data with real-time data augmentation.
    image_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   fill_mode="nearest"
                                   )
    # callbacks
    my_callbacks = [
        # EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ModelCheckpoint(filepath="License_character_recognition.h5",
                        verbose=1, save_weights_only=True),
        LearningRateScheduler(lr_scheduler)
    ]

    # Create model
    print('Creating model...')
    model = mobilev2_base(lr=INIT_LR, train_base=True)

    # Train Model
    print('Training...')
    # ASIDE: if steps_per_epoch is set then do not need to specify batch_size
    # model.fit returns a History istance
    results = model.fit(
        image_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_test, y_test),
        validation_steps=len(X_test) // BATCH_SIZE,
        epochs=EPOCHS, callbacks=my_callbacks,
    )

    fig = plt.figure(figsize=(12, 6))
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    fig.add_subplot(grid[0])
    plt.plot(results.history['accuracy'], label='training accuracy')
    plt.plot(results.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    fig.add_subplot(grid[1])
    plt.plot(results.history['loss'], label='training loss')
    plt.plot(results.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    # Save loss and accuracy plot
    model_num = arg.model_num
    plt.savefig('../img/acc_loss_plot_' + model_num)

    # Save model architecture to JSON
    model_json = model.to_json()
    # model_name = 'mobilenet_char_recog_128in'
    with open("../model/mobilenet_char_recog_128in.json", "w") as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    main()
