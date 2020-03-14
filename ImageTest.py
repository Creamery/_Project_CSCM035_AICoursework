
# load json and create model
from __future__ import division
from keras.models import model_from_json
import os
import numpy as np
import cv2

import Constants


def start():

    # load the model
    json_file = open('fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("fer.h5")
    print("Loaded model from disk")

    # setting image resizing parameters
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # load all images in _images folder
    image_files = load_images(Constants._IMAGES)
    print("Image Loaded")

    # iterate through the files, detect the face, then check emotions based on the model
    for image_file in image_files:

        img_dimensions = image_file.shape
        img_height = img_dimensions[0]

        if img_height > 700:
            image_file = resize_image(image_file, height = 350)

        gray = cv2.cvtColor(image_file, cv2.COLOR_RGB2GRAY)
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face.detectMultiScale(gray, 1.3, 10)

        # detect faces
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha = 0, beta = 1, norm_type = cv2.NORM_L2, dtype = cv2.CV_32F)
            cv2.rectangle(image_file, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # predict emotion
            yhat = loaded_model.predict(cropped_img)
            cv2.putText(image_file, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            print("Emotion: " + labels[int(np.argmax(yhat))])

        cv2.imshow('Emotion', image_file)
        cv2.waitKey()


def load_images(folder):
    images = []
    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)

    return images


def resize_image(image, width = None, height = None, inter = cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized