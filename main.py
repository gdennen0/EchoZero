from Model.main_model import Model
from Control.controller import Control
from Command.command import Command
from message import Log
def main():
    # Initialize an instance of audio_model
    model = Model()
    command = Command(model)
    control = Control(model, command)
    Log.special("Initialized Application")
    control.ingest(path="/Users/gdennen/Projects/Ideas/Audio/3 LETS GO ON THE RUN.mp3") # temp testing

if __name__ == "__main__":
    main()

