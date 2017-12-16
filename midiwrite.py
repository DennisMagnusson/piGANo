from __future__ import print_function
import midi


def write_midi(pattern):
  m = []
  delta = 480/4
  time = 0
  m.append(midi.KeySignatureEvent(tick=0, data=[0, 0]))
  m.append(midi.SetTempoEvent(tick=0, data=[13, 59, 231]))
  m.append(midi.ControlChangeEvent(A

  m.append(midi.ControlChangeEvent(tick=0, channel=1, data=[91, 127]))
  m.append(midi.ControlChangeEvent(tick=0, channel=1, data=[10, 64]))
  m.append(midi.ControlChangeEvent(tick=0, channel=1, data=[7, 100]))
  m.append(midi.ControlChangeEvent(tick=0, channel=2, data=[10, 64]))
  m.append(midi.ControlChangeEvent(tick=0, channel=2, data=[7, 100]))

  for frame in pattern:
    for i in range(len(frame)):
      if frame[i] == 1:
        m.append([NoteOnEvent(tick=time, channel=1, data=[i+20, 100])])


    time += delta

  m.append(midi.EndTrackEvent(tick=0, data=[]))
  m.make_ticks_rel()
  midi.write_midifile(midi.Pattern(foramt=0, resolution=480, tracks=[midi.Trakc[m]]))
