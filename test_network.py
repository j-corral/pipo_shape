#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import dumper
from unidecode import unidecode

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-s", "--silent", required=False, help="enable silent mode")
args = vars(ap.parse_args())

# check silent mode
silent = False
if args['silent']:
	silent = True

# load the image
image = cv2.imread(unidecode(args["image"]))
orig = image.copy()

# load labels
if not silent:
	print("Loading labels.json...")
LIST_WORD = dumper.loadData('labels.json')

if len(LIST_WORD) == 0:
	raise Exception("No labels found into labels.json ! Did you trained the model ?")

if not silent:
	print(LIST_WORD)

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
if not silent:
	print("[INFO] loading network...")
# model = load_model(args["model"])
model = load_model("dataset.model")

# classify the input image
i = 0;
for k in model.predict(image)[0]:
	if not silent:
		print(LIST_WORD[i]["name"] + " : " + str(k))
	LIST_WORD[i]["variable"] = k
	i = i + 1

proba = max(model.predict(image)[0])

# build the label
for word in LIST_WORD:
	if proba == word["variable"]:
		label = word["name"]

label = "{}: {:.2f}%".format(label, proba * 100)

if silent:
	print (label)
else:
	# draw the label on the image
	output = imutils.resize(orig, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

	# show the output image
	cv2.imshow("Output", output)
	cv2.waitKey(0)
