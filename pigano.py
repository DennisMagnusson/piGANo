import GAN
import preprocessor

if __name__ == '__main__':
  g = GAN()
  data = preprocessor.read_dataset("./data", length=-1)
  batch_size = 8
  print("Pre-training")
  g.pre_train(data[:1000], batch_size, epochs=5)
  print("Training")
  g.train(data[1000:], batch_size, epochs=20)
