from Utils.message import Log
import librosa
import numpy as np
from Utils.tools import prompt_selection
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher


class MAOSC():
    def __init__(self, events, ip="127.0.0.1", port=8000):
        self.client = SimpleUDPClient(ip, port)
        Log.info(f"OSC Client initialized at {ip}:{port}")

    def export_onsets(self, event_data):
        for event in event_data.items:
            if event.name == "onset":
                self.client.send_message("/onset", event.data)
                Log.info(f"Sent OSC message with onset at {event.data}")
        

    def set_pool_ints(self):
        # Sets tc pool + sequence pool
        pass

    def events_to_tc(self):
        # Converts time event to time + frames for MA
        pass

    def get_ma_framerate(self):
        # Gets MA framerate from TC pool
        pass

    def get_plugins_list(self):
        # Gets a list of used plugins to find where to store the plugin for get_ma_framework
        pass

    


    # Function to set tc pool + sequence pool
    # Installs plugin to get 

    # User specifies track to apply triggers to, and sequence... or sequence pool?
    # Function stores track
    # U
    # MA3 TIMECODE STRUCTURE: TimecodePool > TrackGroup > Track > Marker
    # SYNTAX: Set Timecode "Intro" "Duration" "55"
    # Store Timecode "Napalm Skies"
    # GOOD TO KNOW KW: 
    # Set: sets values to properties of objects... also used to to transfer properties of objects to the same value as another object...
    #   Used in conjunction with the Property KW
    # Property: Object KW used to set a specific property of an object
    # Set Timecode 102.1.1 "Duration" "2.01"
    # 
    # FLOW OF SYNTAX:#
    # Store TC X.X
    # Assign Sequence Y at TC X.X
    # FOR LOOP FOR EVERY EVENT: 
    #   Set Timecode X "Cursor" "TIME HERE"
    #   Record Timecode X
    #   Go Sequence Y Cue Z
    #   Off Timecode X

    # Leave the option to specify a sequence, specify a sequence etc in a dropdown... on exectution 
    # Make it a part to add... if user didn't add a part, prompt to select a tc / sequence
    # 