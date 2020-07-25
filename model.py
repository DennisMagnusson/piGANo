from __future__ import print_function

import numpy as np

import torch
from torch import nn, optim, tensor

import matplotlib.pyplot as plt

import preprocessor
import midiwrite
import song

datasetino = []


class Discriminator(nn.Module):
  def __init__(self, n_filters=[1,4,8,16,32], filter_sizes=[13,9,5,3,3]):
    super(Discriminator, self).__init__()
    self.convs = []
    self.dropouts = []
    self.convs.append(nn.Conv2d(n_filters[0], n_filters[1], filter_sizes[0], stride=1, padding=6))
    for i in range(1, len(n_filters)-1):
      f = filter_sizes[i]
      self.convs.append(nn.Conv2d(n_filters[i], n_filters[i+1], f, stride=1, padding=int(f/2)))
      self.dropouts.append(nn.Dropout(0.8))

    self.fc1 = nn.Linear(88*64*16, 64)
    self.fc2 = nn.Linear(64, 1)


  def forward(self, x):
    for i in range(len(self.dropouts)):
      x = self.convs[i](x)
      x = self.dropouts[i](x)
      x = nn.LeakyReLU(0.2)(x)

    x = x.reshape(-1, 16*88*64)

    x = self.fc1(x)
    x = nn.Sigmoid()(x)
    x = self.fc2(x)
    x = nn.Sigmoid()(x)

    return x


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.dropout = nn.Dropout(0.5)
    self.fc1 = nn.Linear(10, 22*32*16)#10 = input_size
    #self.upsample = nn.Upsample((88, 64, 32))
    self.upsample = nn.Upsample(scale_factor=4)
    self.bn1 = nn.BatchNorm1d(22*32*16)#10 = input_size

    self.convs = []
    self.convs.append(nn.Conv2d(32, 16, 13, stride=1, padding=6))
    self.convs.append(nn.Conv2d(16, 8, 7, stride=1, padding=3))
    self.convs.append(nn.Conv2d(8, 4, 5, stride=1, padding=2))
    self.convs.append(nn.Conv2d(4, 1, 3, stride=1, padding=1))

    self.bns = []
    self.bns.append(nn.BatchNorm2d(32))
    self.bns.append(nn.BatchNorm2d(16))
    self.bns.append(nn.BatchNorm2d(8))
    self.bns.append(nn.BatchNorm2d(4))

  def forward(self, x):
    x = self.dropout(x)
    x = nn.LeakyReLU(0.2)(self.fc1(x))
    x = self.bn1(x)

    x = torch.reshape(x, (-1, 32, 22, 16))
    x = self.upsample(x)

    for i in range(len(self.convs)):
      #x = self.bns[i](x)
      x = self.convs[i](x)
      x = nn.LeakyReLU(0.2)(x)

    return nn.ReLU()(torch.sign(x))#Poor man's step function

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
    model.add(Conv2d(n_filters[0], filter_sizes[0], input_shape=(88, 64, 1), padding='same'))
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

def print_grayscale(pixels, width=64, height=88):
  pixels = np.transpose(pixels.reshape(88, 64))
  for row in pixels:
    for pixel in row:
      color = 232 + round(pixel*23)
      print('\x1b[48;5;{}m \x1b[0m'.format(int(color)), end="")
    print()

def train(data):
  loss = nn.BCELoss()
  d = Discriminator()
  d_opt = optim.Adam(d.parameters(), lr=0.001)
  g = Generator()
  g_opt = optim.Adam(g.parameters(), lr=0.001)

  batch_size = 32
  data_size = data.shape[0]
  data = np.transpose(data, (0, 3, 1, 2))

  #Pretraining discriminator
  for i in range(10):
    noise = torch.rand(32, 10)
    fake_data = g(noise)
    real_data = torch.Tensor(data[0:32])

    real_outp = d(real_data)
    fake_outp = d(fake_data.detach())

    d_opt.zero_grad()
    fake_loss = loss(real_outp, torch.zeros(32))
    real_loss = loss(fake_outp, torch.ones(32))
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    d_opt.step()

    print(d_loss)

  #Real training
  for ep in range(50):
    for i in range(0, data_size-batch_size, batch_size):
      noise = torch.rand(batch_size, 10)
      fake_data = g(noise)
      real_data = torch.Tensor(data[i:i+batch_size])

      g_opt.zero_grad()
      g_loss = loss(d(fake_data), torch.zeros(batch_size))
      g_loss.backward()
      g_opt.step()

      real_outp = d(real_data)
      fake_outp = d(fake_data.detach())

      d_opt.zero_grad()
      fake_loss = loss(real_outp, torch.zeros(batch_size))
      real_loss = loss(fake_outp, torch.ones(batch_size))
      d_loss = (real_loss + fake_loss) / 2

      d_loss.backward()
      d_opt.step()

      if i % (batch_size*20) == 0:
        print('---LOSSES---')
        print(d_loss)
        print(g_loss)

      if i % (batch_size*200) == 0:
        print_grayscale(fake_data.detach().numpy()[0])

    print('---LOSSES---')
    print(d_loss)
    print(g_loss)
    if ep % 20 == 0:
      print_grayscale(fake_data.detach().numpy()[0])
      pass




  #print_grayscale(y.detach().numpy()[0])
  #print_grayscale(data[0])

if __name__ == '__main__':
  train()
