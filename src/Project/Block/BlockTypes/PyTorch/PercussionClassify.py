from __future__ import print_function
import os
import sys
import librosa
import numpy as np
import librosa.display
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from src.Project.Block.BlockTypes.PyTorch.lib.pytorch_utils import DataLoader, make_pos_encodings, HyperParams, Classifier
from src.Project.Block.BlockTypes.PyTorch.lib.drumclassifier_constants import INSTRUMENT_NAMES
from src.Project.Block.BlockTypes.PyTorch.lib.drumclassifier_utils import transform, DrumClassifier
from src.Utils.message import Log

from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput

sys.path.append("code/")

class PercussionClassify(Block):
    name = "PercussionClassify"
    type = "PercussionClassify"
    def __init__(self):
        super().__init__()
        self.name = "PercussionClassify"
        self.type = "PercussionClassify"

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # self.command.add("load_model", self.load_model)

    
    def process(self, eventData):
        Log.info(f'PercussionClassify processing {len(eventData)} event data items')
        if eventData is None:
            Log.error("EventData is None")
            return None
        drumcl = DrumClassifier(path_to_model= "models/mel_cnn_models/mel_cnn_model_high_v2.model", file_types = None, hop_length = None, clip=None, bs=None, maxseqlen = None, minseqlen=None,pad=None)
        results = None
        results = drumcl.predict_eventData(eventData, format="label")
        Log.info(f"PercussionClassify results: {results}")
        return results
        
    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }  
    
    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        
        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push loaded data to output 
        self.output.push_all(self.data.get_all())    
