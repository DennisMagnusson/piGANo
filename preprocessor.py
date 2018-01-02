from os import listdir
from os.path import isfile

import numpy as np

import midiparse

def read_dataset(directory, length=-1):
  files = list(filter(lambda f: not isfile(f), listdir(directory)))
  
  songs = [midiparse.readfile_midi("data/"+f) for f in files]
  dataset = []
  for song in songs:
    for div in divide_song(song, 64):
      dataset.append(div)
  
  if length != -1:
    dataset = dataset[:length]

  dataset = np.array(dataset)
  l = dataset.shape[0]
  dataset = dataset.reshape(l, 64, 88, 1).swapaxes(1,2)
  return dataset
  
#TODO Test
#TODO What to do with the last part?
def divide_song(song, sequence_len):
  s = []
  for i in range(0, len(song)-sequence_len, sequence_len):
    s.append(song[i:i+sequence_len])

  return s

def next_batch(dataset, index, batch_size):
  for song in dataset:
    if index < len(song):
      break
    else:
      index -= len(song)
    index -= 1
  return False
  #TODO How should I do this?
  #One bar at a time?
  #One sequence_length sequence at a time?
  #One note at a time?
