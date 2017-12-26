import numpy as np

from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten, Dropout, Input

import preprocessor

class GAN:
  def __init__(self):
    self.D = self.create_discriminator()
    self.D.compile(loss='binary_crossentropy', optimizer='Adam')
    self.D.summary()
    self.D.trainable = False
    self.G = self.create_generator()
    self.G.compile(loss='binary_crossentropy', optimizer='Adam')
    self.G.summary()
    inp = Input(shape=(66, 46, 20,))
    #Model(inp, self.D(self.G(inp)))
    self.GAN = Model(inp, self.D(self.G(inp)))
    self.GAN.compile(loss="binary_crossentropy", optimizer="Adam")
    data = preprocessor.read_dataset("./data", length=60)
    self.train(data, 20)

  def train(self, dataset, batch_size):
    dataset_length = dataset.shape[0]
    real_songs = dataset

    input_shape = (20, 66, 46, 20)
    #x = np.concatenate((gen_songs, real_songs), axis=0)
    #perm = np.random.permutation(len(x))
    #x = x[perm]
    #y = y[perm]
    for i in range(0, len(dataset), dataset_length):
      n = self.noise(batch_size, input_shape)
      print(n.shape)
      gen_songs  = self.round(self.G.predict(self.noise(batch_size, input_shape)))#Generate songs
      G_loss = self.D.train_on_batch(gen_songs, np.array([[0]]*batch_size))#Train on fake songs. Returns integer loss in form of an integer.
      D_loss = self.D.train_on_batch(dataset[i:i+batch_size], np.array([[1]]*batch_size))

      print(self.D.predict(gen_songs))

      gen_loss = self.GAN.train_on_batch(self.noise(batch_size, input_shape), np.ones((batch_size, 1)))

      #gan_loss = 1 - self.D.train_on_batch(gen_songs, np.array([0, 1]*batch_size))#TODO Check if this works
      #self.D.train_on_batch(x, y)
      #predictions = self.D.predict(x)
      #self.GAN.train_on_batch(x, predictions)
      #TODO Error.
      #Update weights of G. Somehow.
      print(G_loss)
      print(D_loss)
      print(gen_loss)


  def create_discriminator(self):
    model = Sequential([
      Conv2D(10, (12, 4), input_shape=(88, 64, 1)),
      Dropout(0.3),
      Activation('relu'),
      Conv2D(20, (12, 16)),
      Dropout(0.3),
      Activation('relu'),
      Flatten(),
      Dropout(0.3),
      Dense(1),
      Activation('sigmoid')
    ])

    #inp = Input(shape=(88, 64, 1))
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

    return model

  def round(self, x):
    return np.where(x < 0.5, 0, 1)

  def generate(self):
    songs = self.G.predict(noise(batch_size))
    return songs

  def noise(self, n, shape):
    return np.random.random((n,) + shape[1:])
