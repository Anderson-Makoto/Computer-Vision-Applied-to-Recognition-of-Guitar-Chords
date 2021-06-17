from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import segmentation_models as sm

batch_size = 20
img_height = 64
img_width = 64

def getImagesAndSeparateTrainDataset () :

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "scripts/reduced_classes_chords",
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "scripts/reduced_classes_chords",
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    configureDatasetToBetterPerformance(train_ds, val_ds)

def configureDatasetToBetterPerformance (train_ds, val_ds) :
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    model = createandCompileModel()
    trainningModelAndPredict(train_ds, val_ds, model)

def createandCompileModel () :
    num_classes = 8
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])

    # model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = [sm.metrics.iou_score ,"accuracy"])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print(model.summary())

    return model

def trainningModelAndPredict (train_ds, val_ds, model) :
    epochs=50
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    yVal = tf.concat([y for x, y in val_ds], axis=0)

    yPred = model.predict(val_ds)

    yPred = tf.argmax(yPred, axis=1)

    cr = classification_report(yVal, yPred, target_names = ["DO", "FA", "FA#", "LA", "MI", "RE", "SI", "SOL"])
    # cm = tf.math.confusion_matrix(yVal, yPred)

    print("---------------------------------\n"+str(cr)+"\n-----------------------------------------------")
    getMetrics(history, epochs)

def getMetrics (history, epochs) :
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

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

getImagesAndSeparateTrainDataset()