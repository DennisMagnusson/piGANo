from __future__ import print_function

import numpy as np

#import plaidml.keras
#plaidml.keras.install_backend()

from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten, Dropout, Input, BatchNormalization, UpSampling2D, Reshape, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

import preprocessor
import midiwrite
import song

datasetino = []

class GAN:
  def __init__(self):
    global datasetino
    self.D = self.create_discriminator([(12, 8), (7, 5), (4, 4)], [32, 64, 64], dropout=0.3)
    self.D.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))
    self.D.summary()

    self.G = self.create_generator([(12, 12), (7, 7), (2, 2)], [64, 100, 100])
    self.G.compile(loss='binary_crossentropy', optimizer=Adam(0.001))
    self.G.summary()
    inp = Input(shape=self.input_shape[1:])

    self.set_d_trainable(False)
    self.GAN = Model(inp, self.D(self.G(inp)))
    self.GAN.compile(loss="binary_crossentropy", optimizer=Adam(0.001))

  def plot_gen(self, n_ex=16, dim=(4,4), figsize=(10,10)):
    noise = self.noise(1, self.input_shape)
    generated_images = self.G.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)

  def print_greyscale(self, pixels, width=64, height=88):
    pixels = pixels.reshape(88, 64)
    for row in pixels:
      for pixel in row:
        color = 232 + round(pixel*23)
        print('\x1b[48;5;{}m \x1b[0m'.format(int(color)), end="")
      print()

  def train(self, dataset, batch_size, epochs=1):
    print(dataset.shape)
    dataset_length = dataset.shape[0]
    print(batch_size, dataset_length)
    real_songs = dataset

    for e in range(epochs):
      total_G_loss = 0
      total_D_loss = 0
      total_GAN_loss = 0
      predictions = []
      other_predictions = []
      gen_songs = []
      for i in range(0, dataset_length-batch_size, batch_size):
        noise = self.noise(batch_size, self.input_shape)
        gen_songs = self.G.predict(noise)#Generate songs
        predictions = self.D.predict(gen_songs)
        other_predictions = self.D.predict(dataset[i:i+batch_size])

        #G_loss = 1-self.D.train_on_batch(gen_songs, np.zeros((batch_size, 1)))
        #D_loss = self.D.train_on_batch(dataset[i:i+batch_size], np.ones((batch_size, 1)))

        G_loss = 1-self.D.train_on_batch(gen_songs, 0.3*np.random.random((batch_size, 1)))
        D_loss = self.D.train_on_batch(dataset[i:i+batch_size], 0.7+0.3*np.random.random((batch_size, 1)))


        #noise = self.noise(batch_size, self.input_shape)#Fresh new noise
        #GAN_loss = self.GAN.train_on_batch(noise, np.ones((batch_size, 1)))
        GAN_loss = self.GAN.train_on_batch(noise, 0.7+0.3*np.random.random((batch_size, 1)))


        total_G_loss += G_loss
        total_D_loss += D_loss
        total_GAN_loss += GAN_loss

        if i % 2000 == 0:
          print(i)
          song.print_raw_song(gen_songs[0])
          song.print_grayscale(gen_songs[0])

        #if i == 10000:
          #song.print_grayscale(gen_songs[0])
          #song.print_greyscale(self.round(gen_songs[0]))
          #midiwrite.write_midi(self.round(gen_songs[0]).tolist(), "test.mid")

        print("{}/{} G_loss={}, D_loss={}, GAN_loss={}".format(i, dataset_length, G_loss, D_loss, GAN_loss), end='\r')
     
      print("********************")
      gen_songs = self.G.predict(self.noise(1, self.input_shape))
      self.print_greyscale(gen_songs[0])
      song.print_raw_song(gen_songs[0])

      print(predictions[0:5])
      print(other_predictions[10:15])
      print(e)
      iterations = dataset_length / batch_size
      total_D_loss /= iterations
      total_G_loss /= iterations
      print(total_D_loss, total_G_loss)
      total_GAN_loss /= iterations
      total_D_loss = 1.0/2*(total_D_loss+(1-total_G_loss))
      total_G_loss = G_loss
      print("G_loss= ", total_G_loss)
      print("D_loss= ", total_D_loss)
      print("GAN_loss= ", total_GAN_loss)

  #Give the generator a head start.
  def pre_train(self, dataset, batch_size, epochs=1):
    #self.D.trainable = True
    self.set_d_trainable(True)
    dataset_length = dataset.shape[0]

    for e in range(epochs):
      loss = 0
      sum_loss = 0
      prev = 0
      for i in range(0, dataset_length-batch_size, batch_size):
        prev = loss
        noise = self.noise(batch_size, self.input_shape)
        #loss = self.G.train_on_batch(noise, dataset[i:i+batch_size])
        gen_songs = self.G.predict(noise)
        sum_loss += loss

        x = np.concatenate((dataset[i:i+batch_size], gen_songs))
        y = np.concatenate((np.ones((batch_size, 1)),  np.zeros((batch_size, 1))))
        h = self.D.fit(x, y, shuffle=True, verbose=0, epochs=2)
        d_loss = h.history['loss'][1]

        print(loss, i, "/", dataset_length, "d_loss:", d_loss, end="\r")

      sum_loss /= (dataset_length-batch_size)/batch_size
      if sum_loss < 0.02:
        break
      print("Avg loss= ", sum_loss)
    gen_song = self.G.predict(self.noise(1, self.input_shape))
    song.print_raw_song(gen_song[0])
    #self.D.trainable = False
    self.set_d_trainable(False)

  def set_d_trainable(self, t):
    self.D.trainable = t
    for layer in self.D.layers:
      layer.trainable = t
  
  def create_discriminator(self, filter_sizes, n_filters, dropout=0.6):
    model = Sequential()
    model.add(Conv2D(n_filters[0], filter_sizes[0], input_shape=(88, 64, 1), padding='same'))
    #model.add(AveragePooling2D(pool_size=(2, 2)))#TODO Remove?
    model.add(Dropout(dropout))
    model.add(LeakyReLU(0.2))
    #model.add(Conv2D(n_filters[1], filter_sizes[1], input_shape=(28, 28, 1), padding='same'))

    for i in range(1, len(filter_sizes)):
      model.add(Conv2D(n_filters[i], filter_sizes[i], padding='same'))
      #model.add(AveragePooling2D(pool_size=(2, 2)))
      model.add(Dropout(dropout))
      model.add(LeakyReLU(0.2))
      #model.add(Activation('leakyrelu'))

    #model.add(AveragePooling2D(1, 2))#TODO Doesn't work with this
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model
  
  #TODO make the arguments work properly
  def create_generator(self, filter_sizes, n_filters):
    model = Sequential()
    self.input_shape=(1, n_filters[0])
    model.add(Dropout(0.5, input_shape=self.input_shape[1:]))#TODO Is this good?
    model.add(Dense(22*16*10))
    #model.add(Dense(22*16*10, input_shape=self.input_shape[1:]))
    #model.add(Activation('leakyrelu'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Reshape((22, 16, 10)))
    model.add(UpSampling2D((4, 4)))
    for i in range(1, len(filter_sizes)):
      model.add(Conv2D(n_filters[i], filter_sizes[i], padding='same'))
      model.add(BatchNormalization())
      model.add(LeakyReLU(0.2))
      #model.add(Activation('leakyrelu'))

    model.add(Conv2D(1, (1, 1), padding='same'))
    model.add(Activation('sigmoid'))

    return model

  #TODO Replace with sample of some sort?
  def round(self, x):
    return np.where(x < 0.7, 0, 1)

  def generate(self):
    songs = self.G.predict(noise(batch_size))
    return songs

  def noise(self, n, shape):
    return np.random.normal(0.0, 1.0, (n,) + shape[1:])

