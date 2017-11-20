from os import listdir
from os.path import isfile

import midiparse

def read_dataset(directory):
  files = list(filter(lambda f: isFile(f), listdir(directory)))

  print(files)#Test
  
  songs = [midiparse.readfile_midi(f) for f in files]
  dataset = [divide_song(song) for song in songs]

  return dataset
  

def divide_song(song):
  return False
  #TODO How should I do this?
  #One bar at a time?
  #One sequence_length sequence at a time?
  #One note at a time?
