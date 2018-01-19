from __future__ import print_function
import midi


def write_midi(pattern, filename):
  delta = 480/4

  track0 = midi.Track([
    midi.SmpteOffsetEvent(tick=0, data=[96, 0, 3, 0, 0]),
    midi.TimeSignatureEvent(tick=0, data=[4, 2, 24, 8]),
    midi.KeySignatureEvent(tick=0, data=[0, 0]),
    midi.SetTempoEvent(tick=0, data=[7, 161, 32]),
    midi.EndOfTrackEvent(tick=0, data=[])
  ], tick_relative=False)
  
  m = [
    midi.PortEvent(tick=0, data=[0]),
    midi.ProgramChangeEvent(tick=0, channel=0, data=[0]),
    midi.ControlChangeEvent(tick=0, channel=0, data=[7, 127]),
  ]

  time = 0

  for frame in pattern:
    for i in range(len(frame)):#Should be 88
      if frame[i] == 1:
        m.append(midi.NoteOnEvent(tick=time, channel=0, data=[i+21, 100]))
        m.append(midi.NoteOnEvent(tick=time+delta, channel=0, data=[i+21, 0]))

    time += delta

  m.append(midi.EndOfTrackEvent(tick=time+1, data=[]))
  m = midi.Track(m)

  pattern = midi.Pattern(tracks=[track0, m], tick_relative=False)
  pattern[1].tick_relative = False
  pattern[1].make_ticks_rel()

  midi.write_midifile(midifile=filename, pattern=pattern)
  return pattern
