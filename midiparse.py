from __future__ import print_function
import mydy as midi
from mydy.FileIO import read_midifile

import numpy as np

#Does not use music21. Also way faster
def readfile_midi(filename):
  try:
    pattern = read_midifile(filename)
  except Warning:
    print("Issue with", filename)
    return []
  except TypeError:
    print("error with", filename)
    return []
  
  for i in range(len(pattern)):
    pattern[i] = pattern[i].make_ticks_abs()

  pattern.sort()
  pattern = [list(filter(lambda x: type(x) == midi.Events.NoteOnEvent, p)) for p in pattern]

  time_sig = []

  delta = 480/8#Don't ask why.
  quantize(pattern, delta)
  
  time = 0
  song = []

  maxtick = 0

  for track in pattern:
    if len(track) < 20:
      continue
    
    if track[len(track)-1].tick > maxtick:
      maxtick = track[len(track)-1].tick
 
  for _ in range(0, int(maxtick+delta), int(delta)):
    song.append(np.zeros(88, dtype='uint8'))

  for track in pattern:
    if len(track) < 20:
      continue

    track.sort()

    for event in track:
      if event.data[1] == 0:
        continue

      note = event.data[0] - 21
      if note < 0 or note >= 88:
        print(filename, "has note #", event.data[0]-21)
        return []

      t = int(event.tick/delta)
      song[t][note] = 1
  
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

def quantize(pattern, delta):
  for event in pattern[0]:
      if event.tick % delta != 0:
        if event.tick % delta < delta/2:
          event.tick -= event.tick % delta
        else:
          event.tick += delta - (event.tick % delta)

