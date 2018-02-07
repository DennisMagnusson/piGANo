from model import GAN
from preprocessor import read_dataset

if __name__ == '__main__':
  g = GAN()
  data = read_dataset("./data", length=-1)
  batch_size = 8
  print("Pre-training")
  g.pre_train(data[:1000], batch_size, epochs=5)
  print("Training")
  g.train(data[1000:], batch_size, epochs=20)
