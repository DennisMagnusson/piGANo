from os import listdir
from os.path import isfile

import numpy as np

import midiparse

def read_dataset(directory, length=-1):
  files = list(filter(lambda f: not isfile(f), listdir(directory)))
  songs = []
  for f in files:
    print(len(songs))
    if len(songs)*10 > length:#Just an estimation
      break
    songs.append(midiparse.readfile_midi("data/"+f))
  #songs = [midiparse.readfile_midi("data/"+f) for f in files]
  dataset = []
  for song in songs:
    if length != -1 and len(dataset) > length:
      break

    for div in divide_song(song, 64):
      dataset.append(div)
  

  dataset = np.array(dataset)
  l = dataset.shape[0]
  dataset = dataset.reshape(l, 64, 88, 1).swapaxes(1,2)
  return dataset
  
def divide_song(song, sequence_len):
  s = []
  i = 0
  while i < len(song)-sequence_len:
    while True:
      if np.all(song[i]==0):
        i += 1
      else:
        break
    s.append(song[i:i+sequence_len])
    i += sequence_len

  return s
