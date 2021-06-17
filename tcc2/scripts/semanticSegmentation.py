import cv2
import numpy as np
import os, os.path
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from matplotlib import pyplot as plt


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def openImage (param) :
    if param == 0:
        allFolders = os.listdir(".\scripts\\reduced_classes_chords")
        for folder in allFolders:
            count = 0
            _, _, files = next(os.walk(".\scripts\\reduced_classes_chords\\"+str(folder)))
            for val in range(len(files)):
                if (folder == "DO"):
                    img = cv2.imread(".\scripts\\reduced_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg")
                    # img = cv2.resize(img, (int(img.shape[1]/ 2), int(img.shape[0]/ 2)))
                else:
                    img = cv2.imread(".\scripts\\reduced_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg")
                    # img = cv2.resize(img, (int(img.shape[1]/ 2), int(img.shape[0]/ 2)))
                result_image = preProcessing(img, val)
                cv2.imwrite("./scripts/reduced_knn_classes_chords/" + str(folder) + "/" + str(val) + ".jpg", result_image)
                print(str(count)+"/"+str(len(files) - 1)+" - "+str(folder)+" - img")
                
                count += 1
    elif param == 1:
        segmentationWithCnnSM(param)
    elif param == 2:
        predict()

# ---------------------------------kmeans segmentation--------------------

def preProcessing (image, i) :
    # image = cv2.resize(image, (256, 256))
    image = cv2.bilateralFilter(image, 6, 25, 25)
    imageShape = image
    return segmentationWithKmeans(image, imageShape, i)

def segmentationWithKmeans (image, imageShape, i) :
    image = image.reshape((-1, 3))
    image = np.float32(image)
    K = 20
    attempts = 12
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(image, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((imageShape.shape))
    # cv2.imwrite("./kmeans_segmentation/FA#/" + str(i) + ".jpg", result_image)
    return result_image
    print(i)

# ----------------------------------------cnn segmentation (segmentation_models)-------------------------

def segmentationWithCnnSM (param) :
    backbone = "resnet34"
    height = 128
    width = 128
    trainDataset, maskDataset = loadAllImages(height, width)
    xTrain, xVal, yTrain, yVal = splitTrainAndTest(trainDataset, maskDataset)
    xTrain, xVal = preprocessImageBySegmentationModels(xTrain, xVal, backbone)
    model = defineModel(backbone)
    if (param == 1):
        trainModel(model, xTrain, xVal, yTrain, yVal)
    else:
        predict(model, xTrain, xVal, yTrain, yVal)
    
def preprocessImageBySegmentationModels (xTrain, xVal, backbone) :
    preprocessInput = sm.get_preprocessing(backbone)
    xTrain = preprocessInput(xTrain)
    xVal = preprocessInput(xVal)
    return xTrain, xVal

def loadAllImages (height, width) :
    trainDataset = []
    maskDataset = []

    allFolders = os.listdir(".\scripts\\reduced_classes_chords")
    for folder in allFolders:
        count = 0
        _, _, files = next(os.walk(".\scripts\\reduced_classes_chords\\"+str(folder)))
        for val in range(len(files)):
            if (folder == "DO"):
                img = cv2.imread(".\scripts\\reduced_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (height, width))
                trainDataset.append(img)
            else:
                img = cv2.imread(".\scripts\\reduced_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (height, width))
                trainDataset.append(img)
            print(str(count)+"/"+str(len(files) - 1)+" - "+str(folder)+" - img")
            count += 1
    trainDataset = np.array(trainDataset)

    allFolders = os.listdir(".\scripts\\reduced_masks_chords")
    for folder in allFolders:
        count = 0
        _, _, files = next(os.walk(".\scripts\\reduced_masks_chords\\"+str(folder)))
        for val in range(len(files)):
            img = cv2.imread(".\scripts\\reduced_masks_chords\\"+str(folder)+"\\"+str(val)+".jpg", 0)
            img = cv2.resize(img, (height, width))
            maskDataset.append(img)
            print(str(count)+"/"+str(len(files) - 1)+" - "+str(folder)+" - mask")
            count += 1
    maskDataset = np.array(maskDataset)

    return trainDataset, maskDataset
    
def splitTrainAndTest (trainDataset, maskDataset) :
    xTrain, xVal, yTrain, yVal = train_test_split(trainDataset, maskDataset, test_size=0.33, random_state=42)
    return xTrain, xVal, yTrain, yVal

def defineModel (backbone) :
    model = sm.Linknet(backbone, encoder_weights = "imagenet", encoder_freeze = True)
    inp = Input(shape=(128, 128, 1))
    l1 = Conv2D(3, (1, 1))(inp)
    out = model(l1)
    model = Model(inp, out, name = model.name)
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = [sm.metrics.iou_score ,"accuracy"])
    return model

def trainModel (model, xTrain, xVal, yTrain, yVal) : 
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.dirname("./weights_Unet_resnet34_20Epochs_256p"),
    #                                              save_weights_only=True,
    #                                              verbose=1)
    print("---------------------------------- treinando modelo")
    history = model.fit(
        x = xTrain,
        y = yTrain,
        batch_size = 16,
        epochs = 100,
        verbose = 1,
        validation_data = (xVal, yVal),
    )

    model.save("./weights_Linknet_resnet34_100Epochs_128gray")
    getMetrics(history, 100)
    yPred = model.predict(xVal)
    cm = confusion_matrix(yVal, yPred)
    print(cm)
    
def predict () :
    height = 256
    width = 256
    images = []
    model = tf.keras.models.load_model("./weights_Unet_resnet34_50Epochs_256knn", custom_objects = {"iou_score": sm.metrics.iou_score, "accuracy": "accuracy"})
    allFolders = os.listdir(".\scripts\\reduced_knn_classes_chords")
    for folder in allFolders:
        count = 0
        _, _, files = next(os.walk(".\scripts\\reduced_knn_classes_chords\\"+str(folder)))
        for val in range(len(files)):
            if (folder == "DO"):
                img = cv2.imread(".\scripts\\reduced_knn_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (height, width))
                images.append(img)
            else:
                img = cv2.imread(".\scripts\\reduced_knn_classes_chords\\"+str(folder)+"\\"+str(val)+".jpg", cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (height, width))
                images.append(img)
            img = np.expand_dims(img, axis = 0)
            pred = model.predict(img)
            mask_pred = pred.reshape((height, width, 1))
            mask_pred = cv2.cvtColor(mask_pred, cv2.COLOR_GRAY2BGR)
            # plt.imshow(mask_pred, cmap = "gray")
            plt.imsave("./scripts/deep_learning_segmentation/" + str(folder) + "/" + str(val) + ".jpg", mask_pred, cmap = 'gray')
            # cv2.imwrite("./scripts/deep_learning_segmentation/" + str(folder) + "/" + str(val) + ".jpg", mask_pred)
            print(str(count)+"/"+str(len(files) - 1)+" - "+str(folder)+" - img")
            count += 1
    pred = image

# ----------------------------------------cnn segmentation (tensorflow)-------------------------

def segmentationWithCnnTensorflow () :
    height = 512
    width = 512
    trainDataset, maskDataset = loadAllImages(height, width)
    xTrain, xVal, yTrain, yVal = splitTrainAndTest(trainDataset, maskDataset)
    downStack, upStack = createDownStackAndUpStack(height, width)
    model = createModel(height, width, upStack, downStack)
    model = compileAndTrainModel(model, xTrain, yTrain, xVal, yVal)


def createDownStackAndUpStack (height, width) :
    print("1")
    baseModel = tf.keras.applications.MobileNetV2(input_shape=[height, width, 3], include_top=False)
    layerNames = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [baseModel.get_layer(name).output for name in layerNames]
    downStack = tf.keras.Model(inputs=baseModel.input, outputs=layers)
    downStack.trainable = False

    upStack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    return downStack, upStack

def createModel (height, width, upStack, downStack) :
    print("2")
    last = tf.keras.layers.Conv2DTranspose(
      2, 3, strides=2,
      padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[height, width, 3])
    x = inputs

    skips = downStack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(upStack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def compileAndTrainModel (model, xTrain, yTrain, xVal, yVal) :
    print("3")
    epochs = 20
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        x = xTrain,
        y = yTrain,
        batch_size = 16,
        epochs = epochs,
        verbose = 1,
        validation_data = (xVal, yVal)
    )
    print("4")
    getMetrics(history, epochs)

# def predict (model, xTrain, xVal, yTrain, yVal) :
#     yPred = model.predict(xVal)

#     cm = confusion_matrix(yVal, yPred)
#     print(cm)
#     getMetrics(history, 50)


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

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# run codes
# 0 = knn, 1 = deeplearning(segmentation_models), 2 = predictions(segmentaion_models)
openImage(2)