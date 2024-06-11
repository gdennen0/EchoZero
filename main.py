from Model.main_model import Model
from Control.controller import Control
from Command.command import Command

def main():
    # Initialize an instance of audio_model
    model = Model()
    command = Command(model)
    control = Control(model, command)
    print("Initialized Main Model")

    control.ingest()

if __name__ == "__main__":
    main()

