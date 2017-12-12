from __future__ import print_function
import music21
import midi

"""
#TODO DELETEME
def readfile(filename):
  score = music21.converter.parse(filename, format="midi")
  #This needs to be more dimensions
  l = []
 
  time_sig = score.getTimeSignatures().timeSignature
  bar_duration = time_sig.barDuration

  score = score.quantize((16,))#Quantize to sixteenth
  
  beat = 1
  for note in score.recurse().notesAndRests:
    if beat >= 5:
    if note.beat != beat:
      l.append(EMPTY_FRAME)
      continue
    else:
      l.append(get_notes(note))

    beat += 0.25#I think TODO Be more sure

    #TODO For time. Quantisize by 16th notes. probably. I think
    #REAd here: https://keunwoochoi.wordpress.com/2016/02/23/lstmetallica/

  #For time note.beat might be useful, but what to do about outliers.
  #In elise it's mostly divisible by 1/2, but some are 11/6 and so on.
"""

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

  return song

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

