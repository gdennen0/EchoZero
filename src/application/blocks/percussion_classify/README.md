# Percussion Classify - Modular Classifier System

## Overview

The PercussionClassify block uses a modular architecture that supports multiple classification models as "sub-modules". Each model type is implemented as a separate classifier class, making it easy to add new models without modifying the core block processor.

## Architecture

```
PercussionClassifyBlockProcessor (main block)
    └── PercussionClassifier (abstract base class)
        ├── DrumAudioClassifier (drum-audio-classifier model)
        ├── [Future: OtherClassifier]
        └── [Future: AnotherClassifier]
```

## Adding a New Model Type

To add a new classifier model:

1. **Create a new classifier file** in `percussion_classify/`:
   ```python
   # percussion_classify/my_new_classifier.py
   from . import PercussionClassifier, register_classifier
   
   class MyNewClassifier(PercussionClassifier):
       def get_model_type(self) -> str:
           return "my_new_classifier"
       
       def get_display_name(self) -> str:
           return "My New Classifier"
       
       def get_required_config(self) -> List[str]:
           return ["model_path", "other_param"]
       
       def get_optional_config(self) -> List[str]:
           return ["optional_param"]
       
       def classify_events(self, events: EventDataItem, model_config: Dict) -> EventDataItem:
           # Your classification logic here
           pass
   
   register_classifier(MyNewClassifier)
   ```

2. **Import in `__init__.py`**:
   ```python
   try:
       from . import my_new_classifier
   except ImportError as e:
       Log.debug(f"my_new_classifier not available: {e}")
   ```

3. **That's it!** The classifier will be automatically registered and available.

## Available Models

### drum_audio_classifier

**Source:** https://github.com/aabalke/drum-audio-classifier

**Description:** CNN-based classifier using TensorFlow/Keras. Trained on 2,700+ drum samples.

**Classifies:**
- Kick Drum
- Snare Drum
- Closed Hat Cymbal
- Open Hat Cymbal
- Clap Drum

**Required Configuration:**
- `model_path`: Path to saved TensorFlow model directory

**Optional Configuration:**
- `sample_rate`: Audio sample rate (default: 22050)
- `hop_length`: Hop length for analysis (default: 512)

**Dependencies:**
- `librosa` (for audio processing)
- `tensorflow` (for model loading)

**Usage:**
```python
# In block metadata:
{
    "model_type": "drum_audio_classifier",
    "model_path": "path/to/saved_model"
}
```

## Block Configuration

The PercussionClassify block accepts these metadata parameters:

- `model_type` (required): Identifier for the classifier to use (e.g., "drum_audio_classifier")
- Model-specific parameters: See each classifier's `get_required_config()` and `get_optional_config()`

## Event Requirements

For classification to work, events should have associated audio data. This typically comes from:

1. **DetectOnsets block** with audio slicing enabled
2. Events with `audio_path` or `file_path` in metadata pointing to audio files

Events without audio data will be passed through unchanged with a note in metadata.

## Example Workflow

```
LoadAudio → DetectOnsets → PercussionClassify
```

1. LoadAudio loads an audio file
2. DetectOnsets detects onset times and creates events (with audio clips if configured)
3. PercussionClassify classifies each event's audio clip using the selected model

## Future Models

The modular architecture makes it easy to add:
- PyTorch-based models
- Other TensorFlow models
- Custom classification algorithms
- Real-time classifiers

Just implement the `PercussionClassifier` interface and register it!




