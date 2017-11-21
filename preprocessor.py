from os import listdir
from os.path import isfile

import midiparse

def read_dataset(directory):
  files = list(filter(lambda f: isFile(f), listdir(directory)))

  print(files)#Test
  
  songs = [midiparse.readfile_midi(f) for f in files]
  dataset = [divide_song(song) for song in songs]

  return dataset
  

def next_batch(dataset, index, batch_size):
  for song in dataset:
    if index < len(song):
      break
    else:
      index -= len(song)
    index -= 
  return False
  #TODO How should I do this?
  #One bar at a time?
  #One sequence_length sequence at a time?
  #One note at a time?
