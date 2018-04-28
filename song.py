from __future__ import print_function

def print_song(song):
  for frame in range(song.shape[1]):
    for i in range(song.shape[0]):
      print(int(song[i][frame]), end="")
    print()

def print_grayscale(song):
  for frame in song:
    for note in frame:
      color = 232 + round(pixel*23)
      print('\x1b[48;5;{}m \x1b[0m'.format(int(color)), end="")
    print()


def print_raw_song(song):
  for frame in song.T:
    for note in frame:
      s = "{:2.1f}".format(note)#TODO Clean this up
      if s.endswith("0"):
        s = s[0]+' '
      else:
        s = s[1:]+''
      print(s, end="")
    print()
