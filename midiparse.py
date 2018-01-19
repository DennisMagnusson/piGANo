from __future__ import print_function
import midi

import numpy as np

#Does not use music21. Also way faster
def readfile_midi(filename):
  try:
    pattern = midi.read_midifile(filename)
  except Warning:
    print("Issue with", filename)
    return []

  pattern.make_ticks_abs()
  pattern.sort()
  pattern = [list(filter(lambda x: type(x) == midi.NoteOnEvent, p)) for p in pattern]

  time_sig = []

  time_delta = 480/4 #Don't ask why.
  quantize(pattern, time_delta)
  
  time = 0
  song = []
  for q in range(len(pattern)):
    if len(pattern[q]) < 20:#The lazy way
      continue

    for event in pattern[q]:
      if event.tick > time:
        for i in range(time, event.tick, time_delta):#Fix time
          time += time_delta
          song.append(np.zeros(88, dtype='uint8'))

      if len(event.data) < 2:
        print(event)
        continue
        
      if event.data[1] != 0:
        song.append(np.zeros(88, dtype='uint8'))
        if event.data[0] < 21 or event.data[0] >= 88+21:
          print(filename, "has note #", event.data[0]-21)
          return []
        song[len(song)-1][event.data[0]-21] = 1
  
  trim(song)
  return song

def trim(song):
  for i in song:
    if np.all(i==0):
      song.pop(0)
    else:
      break

  for i in reversed(song):
    if np.all(i==0):
      song.pop()
    else:
      break

def print_song(song):
  for frame in song:
    for note in frame:
      print(note, end="")
    print("")

#Untested
def quantize(pattern, delta):
  for event in pattern[0]:
      if event.tick % delta != 0:
        if event.tick % delta < delta/2:
          event.tick -= event.tick / delta
        else:
          event.tick += delta - event.tick / delta

