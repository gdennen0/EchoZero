from Model.main_model import Model
from Command.command import Command
from message import Log
from listen import main_listen_loop
def main():
    # Initialize an instance of audio_model
    model = Model()
    command = Command(model)
    Log.special("Initialized Application")
    
    main_listen_loop(command) # start listening

if __name__ == "__main__":
    main()

