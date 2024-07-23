from tools import prompt, check_audio_path
from tools import prompt, yes_no_prompt, check_audio_path, create_audio_tensor, create_audio_data
import os
from message import Log
from Command.command_module import CommandModule
import shutil
class Ingest(CommandModule):
    def __init__(self, model, project):
        super().__init__(model=model)
        self.project_dir = project.dir
        self.set_name("Ingest")
        self.add_command("audio", self.add_audio)

    def add_audio(self):
        while True:
            path = str(prompt("Please Enter Path: "))
            if check_audio_path(path):
                abs_path = os.path.abspath(path)
                self.load_audio(abs_path)
                return 
            else:
                Log.error(f"Invalid audio path {path}")
                path = None

    def load_audio(self, audio_file_path, target_sr=None, framerate=None):
        target_sr = target_sr or self.settings['AUDIO_TARGET_SAMPLERATE']
        framerate = framerate or self.settings['AUDIO_FRAMERATE']

        audio_data, _ = create_audio_data(audio_file_path, target_sr)
        t, _ = create_audio_tensor(audio_file_path, target_sr)

        while True:
            name = prompt("Please enter audio object name: ")
            existing_names = [obj.name for obj in self.model.audio.objects]
            if name in existing_names:
                Log.error(f"Name '{name}' already exists in audio objects, please use a unique name")
            else:
                break

        a_path = os.path.join(self.project_dir, "audio", name)
        stems_path = os.path.join(a_path, "stems")
        if os.path.exists(a_path):
            if yes_no_prompt(f"Folder '{a_path}' already exists. Do you want to overwrite it?: "):
                shutil.rmtree(a_path)
                os.makedirs(a_path)
                os.makedirs(stems_path)
            else:
                Log.error(f"Folder '{a_path}' already exists and was not overwritten.")
                return
        else:
            os.makedirs(a_path)
            os.makedirs(stems_path)

        extension = os.path.splitext(audio_file_path)[1]
        shutil.copy(audio_file_path, os.path.join(a_path, name + extension))
        Log.info(f"Copied original audio file to: {a_path}")

        a = self.model.audio.generate_audio_object(self.project_dir, extension, audio_data, t, target_sr, framerate, None, name)
        return a