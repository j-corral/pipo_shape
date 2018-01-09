# USAGE
# python train_network.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import dumper

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

EPOCHS = 100  # Number of epochs to train for
INIT_LR = 1e-3  # Initial learning rate
BS = 10  # Batch Size

LIST_WORD = []

# create labels from dataset path
print("Creating labels...")
index = 0
auto_bs = 0
for path in sorted(os.listdir("dataset")):
    count = len(os.listdir("dataset/" + path))

    if auto_bs == 0 or auto_bs > count:
        auto_bs = count


    LIST_WORD.append({"label": index, "name": path, "variable": 0})
    index = index + 1

BS = auto_bs

if BS < 25 :
    BS = int(BS / 2)

if BS == 0:
    BS = 2

print("batch_size: " + str(BS))


# save labels
print("Saving labels into labels.json...")
dumper.saveData('labels.json', LIST_WORD)


print(LIST_WORD)


# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("dataset")))

# loop over the input images
i = 0
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    print(imagePath)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]


    for word in LIST_WORD:
        if word["name"] == label:
            labels.append(word["label"])
            break

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

NUM_CLASS = len(LIST_WORD)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=NUM_CLASS)
testY = to_categorical(testY, num_classes=NUM_CLASS)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=NUM_CLASS)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# determine loss depending on number of classes
loss_method = "binary_crossentropy"

if NUM_CLASS > 2:
    loss_method = "categorical_crossentropy"

model.compile(loss=loss_method, optimizer=opt, metrics=["accuracy"])

# train the network
# steps_per_epoch=len(trainX) // BS
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX)//BS, epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
# model.save(args["model"])
model.save("dataset.model")
print("model saved.")
