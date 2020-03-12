
import Constants

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2


def start():
    preprocessing()
    initializaion()


def preprocessing():

    warnings.filterwarnings("ignore")
    data = pd.read_csv(Constants._DATASET + '\\fer2013.csv')

    width, height = 48, 48

    datapoints = data['pixels'].tolist()

    # getting features for training
    X = []
    for xseq in datapoints:
        xx = [int(xp) for xp in xseq.split(' ')]
        xx = np.asarray(xx).reshape(width, height)
        X.append(xx.astype('float32'))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)

    # getting labels for training
    y = pd.get_dummies(data['emotion']).as_matrix()

    # storing them using numpy
    np.save('fdataX', X)
    np.save('flabels', y)

    print("Preprocessing Done")
    print("Number of Features: "+str(len(X[0])))
    print("Number of Labels: "+ str(len(y[0])))
    print("Number of examples in dataset:"+str(len(X)))
    print("X,y stored in fdataX.npy and flabels.npy respectively")


def initializaion():
    num_features = 64
    num_labels = 7
    batch_size = 64
    epochs = 100

    x = np.load('./fdataX.npy')
    y = np.load('./flabels.npy')

    x -= np.mean(x, axis = 0)
    x /= np.std(x, axis = 0)

    # for xx in range(10):
    #    plt.figure(xx)
    #    plt.imshow(x[xx].reshape((48, 48)), interpolation='none', cmap='gray')
    # plt.show()

    # splitting into training, validation and testing data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state = 41)

    # saving the test samples to be used later
    np.save('modXtest', X_test)
    np.save('modytest', y_test)

    model = create_cnn(num_features, num_labels)
    compile_model(model, X_train, y_train, batch_size, epochs, X_valid, y_valid)

def create_cnn(num_features, num_labels):
    width, height = 48, 48

    # design the CNN
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size = (3, 3), activation = 'relu', input_shape = (width, height, 1),
                     data_format = 'channels_last', kernel_regularizer = l2(0.01)))
    model.add(Conv2D(num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2 * 2 * 2 * num_features, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation = 'softmax'))

    model.summary()

    return model


def compile_model(model, X_train, y_train, batch_size, epochs, X_valid, y_valid):
    # Compile the model with adam optimixer and categorical crossentropy loss
    model.compile(loss = categorical_crossentropy,
                  optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7),
                  metrics = ['accuracy'])

    # training the model
    model.fit(np.array(X_train), np.array(y_train),
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              validation_data = (np.array(X_valid), np.array(y_valid)),
              shuffle = True)

    # saving the  model to be used later
    fer_json = model.to_json()
    with open("fer.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("fer.h5")
    print("Saved model to disk")



