import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten, Dropout

import preprocessor

class Model:
  def __init__(self):
    self.discriminator = self.create_discriminator()
    self.generator = self.create_generator()
    data = preprocessor.read_dataset("./data", length=20)
    self.train(data, 10)

  def train(self, dataset, batch_size):
    dataset_length = dataset.shape[0]
    real_songs = dataset
    gen_songs  = self.round(self.generator.predict(self.noise(dataset_length, self.generator.input_shape)))
    print(gen_songs.shape)
    print(real_songs.shape)
    #gen_songs = self.noise(dataset_length)
    print(gen_songs.shape)
    x = np.concatenate((gen_songs, real_songs), axis=0)
    y = np.concatenate(([[1, 0]]*dataset_length, [[0, 1]]*dataset_length), axis=0)#[1, 0] -> computer, [0, 1] -> human
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    self.discriminator.fit(x, y, batch_size=2*batch_size)
    predictions = self.discriminator.predict(x)
    #TODO Error.
    #Update weights of generator. Somehow.
    print(predictions)


  def create_discriminator(self):
    model = Sequential([
      Conv2D(10, (12, 4), input_shape=(88, 64, 1)),
      Dropout(0.3),
      Activation('relu'),
      Conv2D(20, (12, 16)),
      Dropout(0.3),
      Activation('relu'),
      Flatten(),
      Dense(20),
      Activation('linear'),
      Dropout(0.3),
      Dense(2),
      Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()
    return model
    
  def create_generator(self):
    model = Sequential([#TODO Add batch norm
      Conv2DTranspose(20, (12, 16), input_shape=(66, 46, 20)),#FIXME dims seem to reduce with 7 for every layer.
      Activation('relu'),
      Conv2DTranspose(10, (12, 4)),
      Activation('relu'),
      Conv2DTranspose(1, (1, 1)),#Good enough
      Activation('sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='Adam')
    model.summary()
    return model

  def round(self, x):
    return np.where(x < 0.5, 0, 1)

  def generate(self):
    songs = self.generator.predict(noise(batch_size))
    return songs


  def discriminate(self):
    self.generator.predict()


  def noise(self, n, shape):
    return np.random.random((n,) + shape[1:])
