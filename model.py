from __future__ import print_function

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten, Dropout, Input, BatchNormalization
from keras.optimizers import Adam, RMSprop

import preprocessor

class GAN:
  def __init__(self):
    optimizer = RMSprop(0.001)
    self.D = self.create_discriminator()
    self.D.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.D.summary()
    self.G = self.create_generator()
    self.G.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.G.summary()
    inp = Input(shape=(43, 15, 10,))
    #Model(inp, self.D(self.G(inp)))

    self.D.trainable = False
    self.GAN = Model(inp, self.D(self.G(inp)))
    self.GAN.compile(loss="binary_crossentropy", optimizer=RMSprop(0.1))
    data = preprocessor.read_dataset("./data", length=-1)
    self.train(data, 60, epochs=20)

  def train(self, dataset, batch_size, epochs=1):
    print(dataset.shape)
    dataset_length = dataset.shape[0]
    print(batch_size, dataset_length)
    real_songs = dataset

    input_shape = (batch_size, 43, 15, 10)#TODO Do this better

    for e in range(epochs):
      G_loss = 0
      D_loss = 0
      GAN_loss = 0
      predictions = []
      other_predictions = []
      gen_songs = []
      for i in range(0, dataset_length-batch_size, batch_size):
        noise = self.noise(batch_size, input_shape)
        gen_songs  = self.round(self.G.predict(noise))#Generate songs
        predictions = self.D.predict(gen_songs)
        other_predictions = self.D.predict(dataset[i:i+batch_size])

        G_loss += self.D.train_on_batch(gen_songs, np.zeros((batch_size, 1)))
        D_loss += self.D.train_on_batch(dataset[i:i+batch_size], np.ones((batch_size, 1)))

        GAN_loss += self.GAN.train_on_batch(noise, predictions)
      
      self.print_song(gen_songs[0])
      print("********************")
      print(predictions[0:5])
      print(other_predictions[10:15])
      print(e)
      iterations = dataset_length / batch_size
      D_loss /= iterations
      G_loss /= iterations
      print(D_loss, G_loss)
      GAN_loss /= iterations
      D_loss = 1.0/2*(D_loss+G_loss)
      G_loss = 1-G_loss
      print("G_loss= ", G_loss)
      print("D_loss= ", D_loss)
      print("GAN_loss= ", GAN_loss)


  def create_discriminator(self):
    model = Sequential([
      Conv2D(1, (12, 4), input_shape=(88, 64, 1)),
      Dropout(0.6),
      Activation('linear'),
      Flatten(),
      Dropout(0.6),
      Dense(1),
      Activation('sigmoid')
    ])

    #inp = Input(shape=(88, 64, 1))
    return model
    
  def create_generator(self):
    model = Sequential([#TODO Add batch norm
      Conv2DTranspose(10, (12, 4), input_shape=(43, 15, 10)),#FIXME dims seem to reduce with n-1 for every layer.
      Activation('linear'),
      BatchNormalization(),
      Conv2DTranspose(5, (12, 16)),
      Activation('linear'),
      BatchNormalization(),
      Conv2DTranspose(1, (24, 32)),#Good enough
      Activation('sigmoid')
    ])

    return model

  def round(self, x):
    return np.where(x < 0.5, 0, 1)

  def generate(self):
    songs = self.G.predict(noise(batch_size))
    return songs

  def noise(self, n, shape):
    return np.random.random((n,) + shape[1:])

  def print_song(self, song):
    for frame in song:
      for i in frame:
        print(int(i), end="")
      print()
