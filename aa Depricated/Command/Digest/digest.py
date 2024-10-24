from Model.event_pool import EventPool
from Model.event import Event
from message import Log
from tools import prompt_selection
from .Analyze.analyze import Analyze
from .PreProcess.pre_process import PreProcess
from Command.command_module import CommandModule
from Command.Digest.Transform.transform import Transform


class Digest(CommandModule):
    def __init__(self, model, settings):
        super().__init__(model=model)
        self.settings = settings
        self.set_name("Digest")
        self.preprocess = PreProcess(self.settings)
        self.analyze = Analyze(self.model)
        self.transform = Transform(self.model)
        self.add_sub_module(self.preprocess)
        self.add_sub_module(self.analyze)
        self.add_sub_module(self.transform)

    def results(self):
        Log.list("Analyzation results", self.results, atrib="data")
