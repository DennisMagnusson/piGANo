import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten

import preprocessor

def step(x):
  return 0 if x < 0 else 1

class Model:
  def __init__(self):
    self.discriminator = self.create_discriminator()
    #self.generator = self.create_generator()
    data = preprocessor.read_dataset("./data")
    self.train(data, 10)

  def train(self, dataset, batch_size):
    dataset_length = dataset.shape[3]
    #positives = self.generator.fit()
    real_songs = dataset
    #gen_songs  = self.generator.predict(self.noise(dataset_length))#TODO Uncomment
    gen_songs = self.noise(dataset_length)
    #TODO Shuffle?
    print(gen_songs.shape)
    x = np.concatenate((gen_songs, real_songs), axis=3)
    y = np.concatenate(([[1, 0]]*dataset_length, [[0, 1]]*dataset_length), axis=0)#[1, 0] -> computer, [0, 1] -> human
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    self.discriminator.fit(x, y, batch_size=2*batch_size)
    predictions = self.discriminator.predict(x)
    print(predictions)


  def create_discriminator(self):
    model = Sequential([
      Conv2D(10, (12, 4), input_shape=(88, 64, 1)),
      Activation('relu'),
      Conv2D(20, (32, 32)),
      Activation('relu'),
      Flatten(),
      Dense(20),
      Activation('linear'),
      Dense(2),
      Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    model.summary()
    return model
    
  def create_generator(self):
    model = Sequential([
      Conv2DTranspose(10, (32, 32), input_shape=(88, 64, 1)),
      Activation('relu'),
      Conv2DTranspose(10, (32, 32)),
      Activation('relu'),
      Conv2DTranspose(10, (32, 32)),
      Activation(step)
    ])

    model.compile(loss='binary_crossentropy', optimizer='Adam')
    model.summary()
    return model

  def generate(self):
    songs = self.generator.predict(noise(batch_size))
    return songs


  def discriminate(self):
    self.generator.predict()


  def noise(self, n):
    return np.random.random((n, 88, 64, 1))
