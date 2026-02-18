# EchoZero Block Visual Guide

Complete reference for all block types in the Qt GUI.

## Block Colors and Categories

### Input/Output Blocks  

**LoadAudio** - Blue (80, 120, 200)
- Loads audio files (WAV, MP3, FLAC)
- Output: `audio` (Audio)
- Use: Start of most processing chains

**ExportAudio** - Teal (100, 200, 150)
- Exports audio to file
- Input: `audio` (Audio)
- Use: Save processed audio


### Analysis Blocks 

**DetectOnsets** - Orange (220, 140, 80)
- Detects onset times in audio
- Input: `audio` (Audio)
- Output: `events` (Event)
- Use: Find hit points, drum hits, note starts


---

### Note Extraction Blocks 

**TranscribeNote** (NoteExtractorBasicPitch) - Light Green (140, 200, 100)
- Extracts musical notes using ML
- Input: `audio` (Audio)
- Output: `events` (Event with notes)
- Use: High-quality note transcription

**NoteExtractorBasicPitch** - Green (150, 200, 100)
- ML-based note extraction
- Input: `audio` (Audio)
- Output: `events` (Event)
- Use: Polyphonic transcription

**TranscribeLib** (NoteExtractorLibrosa) - Lime Green (160, 200, 120)
- Librosa-based note extraction
- Input: `audio` (Audio)
- Output: `events` (Event)
- Use: Fast monophonic transcription

**NoteExtractorLibrosa** - Olive Green (150, 180, 100)
- Librosa note extraction
- Input: `audio` (Audio)
- Output: `events` (Event)
- Use: Lightweight transcription

---

### Processing Blocks 

**Separator** - Purple (180, 100, 180)
- Demucs source separation
- Input: `audio` (Audio)
- Outputs: `drums`, `bass`, `other`, `vocals` (Audio)
- Use: Separate stems from mix

**SeparatorBlock** - Pink-Purple (200, 100, 150)
- Alternative separator
- Input: `audio` (Audio)
- Multiple audio outputs
- Use: Stem separation

---

### Utility Blocks  

**CommandSequencer** - Yellow (200, 200, 80)
- Executes command sequences
- Input: Various
- Output: Various
- Use: Workflow automation

**Editor** - Gray (150, 150, 150)
- Interactive editor block
- Various inputs/outputs
- Use: Manual editing

**EditorV2** - Light Gray (160, 160, 160)
- Updated editor block
- Various inputs/outputs
- Use: Enhanced editing

---

## Common Block Workflows

### Workflow 1: Simple Onset Detection
```
LoadAudio (Blue)
    ↓ audio
DetectOnsets (Orange)
    ↓ events
Editor (Gray) or ExportAudio (Teal)
```

### Workflow 2: Note Transcription
```
LoadAudio (Blue)
    ↓ audio
TranscribeNote (Green)
    ↓ events
Editor (Gray) or ExportAudio (Teal)
```

### Workflow 3: Stem Separation & Analysis
```
LoadAudio (Blue)
    ↓ audio
Separator (Purple)
    ↓ drums, bass, vocals, other
[4x] TranscribeNote (Green)
    ↓ events
[4x] Editor (Gray)
```

### Workflow 4: Drum Classification
```
LoadAudio (Blue)
    ↓ audio
DetectOnsets (Orange)
    ↓ events
DrumClassify (Brown-Orange)
    ↓ events (with kick/snare/hihat labels)
Editor (Gray) or ExportAudio (Teal)
```

---

## Visual Identification Tips

### By Color Family:
- **Blue tones** = Input/Output operations
- **Orange tones** = Analysis operations
- **Green tones** = Note/pitch extraction
- **Purple tones** = Audio processing/effects
- **Yellow/Gray tones** = Utility/automation

### By Position in Chain:
- **Start**: Usually LoadAudio (Blue)
- **Middle**: Analysis (Orange), Processing (Purple), or Extraction (Green)
- **End**: Usually Editor (Gray) or Export (Teal/Green)

### Quick Reference:
- Need to **load audio**? → Blue block (LoadAudio)
- Need **onsets**? → Orange block (DetectOnsets)
- Need **notes**? → Green block (TranscribeNote/NoteExtractor)
- Need **stems**? → Purple block (Separator)
- Need to **edit**? → Gray block (Editor)
- Need to **save**? → Teal block (ExportAudio)

---

## Port Types

All blocks have typed ports:

- **Audio** (Audio signal data)
  - Sample rate, channels, data
  - Used by audio processing blocks

- **Event** (Time-based events)
  - Time, duration, classification, metadata
  - Used by analysis and visualization blocks

- **Data** (Generic data)
  - Custom data types
  - Used by utility blocks

Ports are color-coded:
- **Input ports** (left side): Green circles
- **Output ports** (right side): Red circles

---

## Node Editor Features

### Selection:
- **Click** block to select
- **Shift+Click** for multi-select (future)
- **Drag** selected blocks to move

### Context Menu (Right-click):
- Delete Block
- Rename Block
- Properties

### Keyboard:
- **Delete** key removes selected block (future)
- **Ctrl+Z** for undo (future)

### Visual Feedback:
- **Selected**: Blue border around block
- **Hover**: Slight highlight (future)
- **Connected**: Lines between output/input ports

---

## Tips

1. **Color Coding**: Use colors to quickly identify block types
2. **Block Names**: Give descriptive names (e.g., "drums_audio", "vocal_notes")
3. **Auto-Layout**: Use "Auto Layout" button for clean arrangement
4. **Properties Panel**: Click blocks to see detailed info
5. **Event Editor**: Use Ctrl+E to see event data in timeline

---

## All Block Types at a Glance

```
 LoadAudio           DetectOnsets        TranscribeNote
 ExportAudio         DrumClassify        NoteExtractorBasicPitch
 Separator           TranscribeLib
 SeparatorBlock      NoteExtractorLibrosa
 CommandSequencer    Editor              EditorV2
```

Enjoy visual block editing in EchoZero! 

