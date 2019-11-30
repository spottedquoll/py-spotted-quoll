#!/usr/bin/python

import vlc
import os
import time
from mutagen.mp3 import MP3
from random import random

# options
alarm_length_min = 3  # min
alarm_length_secs = alarm_length_min*60

media_dir = os.environ['media_dir']
f_name = media_dir + 'Tycho - Past is Prologue.mp3'

print('Reading ' + f_name)
audio = MP3(f_name)
track_length = audio.info.length

start = min(random()*track_length, track_length-alarm_length_secs)
assert start > 0

options = 'start-time=' + str(start)
player = vlc.MediaPlayer(f_name, options)
player.play()

time.sleep(alarm_length_secs)
