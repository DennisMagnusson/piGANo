import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Flatten

def step(x):
  return 0 if x < 0 else 1

class Model:
  def __init__(self):
    self.discriminator = self.create_discriminator()
    #self.generator = self.create_generator()
    self.train([], 10)


  def train(self, dataset, batch_size):
    #positives = self.generator.fit()
    positives = self.noise(batch_size)
    negatives = self.noise(batch_size)#TODO FIXME Replace with training data
    #TODO Shuffle?
    x = np.concatenate((positives, negatives), axis=0)
    y = np.concatenate(([[1, 0]]*batch_size, [[0, 1]]*batch_size), axis=0)
    
    self.discriminator.fit(x, y)
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

