from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranpose

class Model:
  def __init__(self):
    self.discriminator = create_discriminator()
    self.generator = create_generator()

    self.gan = self.discriminator(self.generator())#TODO


  def train(dataset):
    self.disciminator.fit()
    self.generator.fit()

  def create_discriminator():
    model = models.Sequential([
      Conv2D((32, 32), input_dim=(88, 64)),
      Activation('relu'),
      Conv2D((32, 32)),
      Activation('relu')
      Dense(2),
      Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    return model
    
  def create_generator():
    model = models.Sequential([
      Conv2DTranspose((32, 32), input_dim=(88, 64)),
      Activation('relu'),
      Conv2DTranspose((32, 32)),
      Activation('relu')
    ])

    model.compile(loss='binary_crossentropy', optimizer='Adam')
    return model

  def generate():
    self.generator.predict()


  def discriminate():
    self.generator.predict()


