# DrumClassify Block

This block uses a pre-trained Convolutional Neural Network (CNN) model from the [drum-audio-classifier](https://github.com/aabalke33/drum-audio-classifier) repository to classify percussion samples into different drum types.

## Overview

The DrumClassify block takes audio samples (typically percussion hits) as input and classifies them into one of the following categories:
- Clap
- Closed Hat (Hi-hat)
- Kick Drum
- Open Hat (Hi-hat)
- Snare Drum

This directly matches the categories used in the original drum-audio-classifier repository.

## Implementation Details

The block processes audio data following the exact method from the original repository:
1. Loading and resampling audio to 22050Hz
2. Trimming silence with a top_db of 50
3. Converting to mel-spectrograms
4. Processing the spectrograms to match the model's expected format (128 frequency bins × 100 time steps × 3 channels)
5. Running the preprocessed data through the TensorFlow CNN model
6. Mapping the model's output to drum type labels

## Usage

The block expects input in the form of EventData containing audio samples. It outputs EventData with classified drum samples, with each item's classification property set to the detected drum type.

## Model Information

The model was trained on over 2,700 percussion samples from H3 Music Corp and has been optimized for recognizing common drum types. As noted in the original repository, the model can classify:
- Kick Drum
- Snare Drum
- Closed Hat Cymbal
- Open Hat Cymbal
- Clap Drum samples

The original model and training methodology can be found at [aabalke33/drum-audio-classifier](https://github.com/aabalke33/drum-audio-classifier). 