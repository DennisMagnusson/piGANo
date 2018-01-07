from __future__ import print_function

import numpy as np

import plaidml.keras
plaidml.keras.install_backend()

from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten, Dropout, Input, BatchNormalization, UpSampling2D, Reshape
from keras.optimizers import Adam, RMSprop

import preprocessor

datasetino = []

class GAN:
  def __init__(self):
    global datasetino
    self.D = self.create_discriminator([(12, 12), (6, 6), (4, 4)], [1, 5, 10], dropout=0.6)
    self.D.compile(loss='binary_crossentropy', optimizer=RMSprop(0.001))
    self.D.summary()
    self.G = self.create_generator([(12, 12), (12, 12), (12, 12)], [50, 25, 1])
    self.G.compile(loss='binary_crossentropy', optimizer=RMSprop(0.0005))
    self.G.summary()
    inp = Input(shape=self.input_shape[1:])

    self.D.trainable = False
    self.GAN = Model(inp, self.D(self.G(inp)))
    self.GAN.compile(loss="binary_crossentropy", optimizer=RMSprop(0.1))

  def run(self):
    data = preprocessor.read_dataset("./data", length=-1)
    datasetino = data
    batch_size = 10
    print("pre-training")
    self.pre_train(data[:1000], batch_size, epochs=2)
    self.train(data[1000:], batch_size, epochs=20)

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
        gen_songs = self.round(gen_songs)
        predictions = self.D.predict(gen_songs)
        other_predictions = self.D.predict(dataset[i:i+batch_size])

        G_loss = self.D.train_on_batch(gen_songs, np.zeros((batch_size, 1)))
        D_loss = self.D.train_on_batch(dataset[i:i+batch_size], np.ones((batch_size, 1)))

        GAN_loss = self.GAN.train_on_batch(noise, predictions)

        total_G_loss += G_loss
        total_D_loss += D_loss
        total_GAN_loss += GAN_loss

        if i % 1500 == 0:
          self.print_raw_song(gen_songs[0])

        print("{}/{} G_loss={}, D_loss={}, GAN_loss={}".format(i, dataset_length, G_loss, D_loss, GAN_loss), end='\r')
     
      print("********************")
      print(predictions[0:5])
      print(other_predictions[10:15])
      print(e)
      iterations = dataset_length / batch_size
      total_D_loss /= iterations
      total_G_loss /= iterations
      print(total_D_loss, total_G_loss)
      total_GAN_loss /= iterations
      total_D_loss = 1.0/2*(total_D_loss+total_G_loss)
      total_G_loss = 1-G_loss
      print("G_loss= ", total_G_loss)
      print("D_loss= ", total_D_loss)
      print("GAN_loss= ", total_GAN_loss)

  #Give the generator a head start.
  def pre_train(self, dataset, batch_size, epochs=1):
    dataset_length = dataset.shape[0]

    for e in range(epochs):
      loss = 0
      prev = 0
      for i in range(0, dataset_length-batch_size, batch_size):
        prev = loss
        loss = self.G.train_on_batch(self.noise(batch_size, self.input_shape), dataset[i:i+batch_size])
        print(loss, i, "/", dataset_length, "delta:", loss-prev, end="\r")
    
      print("Last loss= ", loss)
    song = self.G.predict(self.noise(1, self.input_shape))
    self.print_raw_song(song[0])

  #XXX this prints sideways.
  def print_raw_song(self, song):
    for part in song:
      for note in range(len(part)):
        s = "%.1f"%part[note]
        if s.endswith("0"):
          s = s[0]+'  '
        else:
          s = s[1:]+' '
        print(s, end="")
      print("")

  def create_discriminator(self, filter_sizes, n_filters, dropout=0.6):
    model = Sequential()
    model.add(Conv2D(n_filters[1], filter_sizes[1], input_shape=(88, 64, 1)))

    for i in range(1, len(filter_sizes)):
      model.add(Conv2D(n_filters[i], filter_sizes[i]))
      model.add(Dropout(dropout))
      model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model
    
  def create_generator(self, filter_sizes, n_filters):
    model = Sequential()
    self.input_shape=(1, 100)
    model.add(Dense(22*16*100, input_shape=self.input_shape[1:]))
    model.add(Reshape((22, 16, 100)))
    #model.add(UpSampling2D((2, 2), input_shape=self.input_shape[1:]))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(50, (12, 12), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(25, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(1, (1, 1), padding='same'))
    model.add(Activation('sigmoid'))

    """
    input_dim1 = 88
    input_dim2 = 64
    input_dim3 = n_filters[0]
    for s in filter_sizes:
      input_dim1 -= s[0]-1
      input_dim2 -= s[1]-1

    self.input_shape = (input_dim1, input_dim2, input_dim3)

    model = Sequential()
    model.add(Conv2DTranspose(n_filters[0], filter_sizes[0], input_shape=self.input_shape))
    model.add(Activation("linear"))
    model.add(BatchNormalization())
    for i in range(1, len(filter_sizes)-1):
      model.add(Conv2DTranspose(n_filters[i], filter_sizes[i]))
      model.add(Activation("linear"))
      model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, (filter_sizes[len(filter_sizes)-1])))
    model.add(Activation('sigmoid'))
    """

    return model

  #TODO Replace with sample of some sort?
  def round(self, x):
    return np.where(x < 0.2, 0, 1)

  def generate(self):
    songs = self.G.predict(noise(batch_size))
    return songs

  def noise(self, n, shape):
    return np.random.random((n,) + shape[1:])

  def print_song(self, song):
    for frame in range(song.shape[1]):#64
      for i in range(song.shape[0]):#88
        print(int(song[i][frame]), end="")
      print()


"""
 f = ('function (I[{input_dims_str}], K[{ker_dims_str}]) ' + '-> (O) {{\n{padding_str}\n' +
         '  O[{out_idx_str} : {out_dims_str}] = +(I[{input_idx_str}]*K[{ker_idx_str}]);\n}}')
            .format(**{
             'input_dims_str': input_dims_str,
             'ker_dims_str': ker_dims_str,
             'out_idx_str': out_idx_str,
             'out_dims_str': out_dims_str,
             'input_idx_str': input_idx_str,
             'ker_idx_str': ker_idx_str,
             'padding_str': padding_str
})
"""
