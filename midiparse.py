import music21
import midi

EMPTY_FRAME = [0]*88

#TODO DELETEME
def readfile(filename):
  score = music21.converter.parse(filename, format="midi")
  #This needs to be more dimensions
  l = []
 
  time_sig = score.getTimeSignatures().timeSignature.
  bar_duration = time_sig.barDuration

  score = score.quantize((16,))#Quantize to sixteenth
  
  beat = 1
  for note in score.recurse().notesAndRests:
    if beat >= 5
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

#TODO Needs testing
#Does not use music21. Also way faster
def readfile_midi(filename):
  pattern = midi.read_midifile(filename)
  pattern.make_ticks_abs()
  pattern.sort()
  filter_notes()

  time_sig = []

  time_delta = 480/4 #Don't ask why.
  quantize(pattern, time_delta)
  
  time = 0
  song = []
  for event in pattern[0]:
    if event.tick != time:
      for i in range(time, event.tick, delta):#Fix time
        time += (event.tick - time) / delta
        song.append(EMPTY_FRAME)
      
      if event.data[1] != 0:
        song.append(EMPTY_FRAME)
        song[len(song)-1][event.data[0]-21] = 1

  return song


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
 for i in range(len(pattern[0])):
   if type(pattern[0][i]) != midi.NoteOnEvent:
     pattern[0].pop(i)

def get_notes(note):
  frame = EMPTY_FRAME
  if note.isChord:
    for pitch in note.pitches:
      frame[pitch.midi-21] = 1
  elif note.isNote:
    frame[note.pitch.midi-21] = 1

  return frame

