from __future__ import print_function
import midi

#Does not use music21. Also way faster
def readfile_midi(filename):
  pattern = midi.read_midifile(filename)
  pattern.make_ticks_abs()
  pattern.sort()
  filter_notes(pattern)

  time_sig = []

  time_delta = 480/4 #Don't ask why.
  quantize(pattern, time_delta)
  
  time = 0
  song = []
  for event in pattern[0]:
    if event.tick > time:
      for i in range(time, event.tick, time_delta):#Fix time
        #time += (event.tick - time) / time_delta
        time += time_delta
        song.append([0]*88)
      
    if event.data[1] != 0:
      song.append([0]*88)
      song[len(song)-1][event.data[0]-21] = 1

  trim(song)
  return song

def trim(song):
  for i in song:
    if i == [0]*88:
      song.pop(0)
    else:
      break

  for i in reversed(song):
    if i == [0]*88:
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

#Untested
def filter_notes(pattern):
  i = 0
 
  #for i in range(len(pattern[0])):
  while i < len(pattern[0]):
    if type(pattern[0][i]) != midi.NoteOnEvent:
      pattern[0].pop(i)
      i -= 1
    i += 1

#TODO DELETEME
def get_notes(note):
  frame = [0]*88
  if note.isChord:
    for pitch in note.pitches:
      frame[pitch.midi-21] = 1
  elif note.isNote:
    frame[note.pitch.midi-21] = 1

  return frame

