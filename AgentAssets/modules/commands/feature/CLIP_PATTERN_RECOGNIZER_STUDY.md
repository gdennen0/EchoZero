# Clip Pattern Recognizer - Research Study

## Overview

This document studies approaches for implementing a **Clip Pattern Recognizer** block that identifies and predicts percussion event patterns from messy EventData. The block should:

- Detect timing patterns in percussion EventData items
- Identify similar-sounding events (audio similarity)
- Rebuild/predict patterns as events are collected incrementally
- Handle noisy/messy data from real-world audio processing

## EchoZero Context

### Current EventData Structure

```python
class Event:
    time: float              # Event time in seconds
    classification: str      # e.g., "kick", "snare", "hihat"
    duration: float          # Event duration in seconds
    metadata: Dict[str, Any] # Optional metadata (could include audio features)
```

### Existing Capabilities

- **librosa**: Onset detection, tempo estimation, audio analysis
- **PyTorch/TensorFlow**: ML models for classification
- **scikit-learn**: Machine learning utilities
- **numpy/scipy**: Signal processing
- **Percussion classification**: Already implemented (TensorFlow/PyTorch classifiers)

### Block Architecture

Blocks implement `BlockProcessor` interface:
- `can_process(block) -> bool`
- `process(block, inputs, metadata) -> Dict[str, DataItem]`
- `get_block_type() -> str`

Input: `EventDataItem` with percussion events
Output: `EventDataItem` with recognized patterns and predicted events

---

## Five Approaches to Pattern Recognition

### 1. Template Matching with Enhanced Recognition (Masataka Goto)

**Project**: Sound Source Separation for Percussion Instruments  
**URL**: https://staff.aist.go.jp/m.goto/PROJ/sss.html

**Approach**:
- Enhanced template matching technique
- Extracts instrument type, onset time, and loudness from polyphonic audio
- Recognizes individual percussion sounds in mixtures
- Handles varying loudness levels
- Detects onset times for 9 types of drum instruments

**Key Techniques**:
- Template matching with multiple templates per instrument
- Loudness normalization
- Onset detection with temporal alignment
- Multi-instrument separation

**Applicability to EchoZero**:
- ✅ Can work with EventData (already has onset times)
- ✅ Classification already available (can use as template labels)
- ✅ Could extract audio clips from events and match templates
- ⚠️ Requires audio data, not just EventData (need AudioDataItem input)

**Libraries/Implementation**:
- librosa for template matching (already in project)
- scipy.signal for correlation matching
- numpy for feature extraction

**Pros**:
- Well-established technique
- Works with existing EchoZero audio processing
- Can leverage existing classification results

**Cons**:
- Requires audio data, not just events
- Template matching can be computationally expensive
- May need user-provided templates

---

### 2. Hidden Markov Models (HMM) for Sequence Prediction

**Project**: Discovery of Syllabic Percussion Patterns in Tabla Solo Recordings  
**URL**: https://compmusic.upf.edu/ismir-2015-tabla

**Approach**:
- Uses syllable-level Hidden Markov Models for transcription
- Rough Longest Common Subsequence (RLCS) for pattern discovery
- Builds frequent syllabic patterns from audio + expert scores
- Handles complex rhythmic structures

**Key Techniques**:
- HMM for sequence modeling
- RLCS for robust pattern matching (handles variations)
- Pattern frequency analysis
- State-based prediction

**Applicability to EchoZero**:
- ✅ Perfect for EventData sequences (time + classification)
- ✅ Can predict next events based on pattern
- ✅ Handles noisy data with RLCS
- ✅ Online learning possible (incremental HMM updates)
- ✅ Works with classification strings (kick, snare, etc.)

**Libraries/Implementation**:
- `hmmlearn` (Python HMM library)
- Custom RLCS implementation (or use `difflib.SequenceMatcher`)
- numpy for sequence processing

**Pros**:
- Excellent for temporal sequence prediction
- Handles variations and noise
- Can learn patterns incrementally
- Works with EventData directly (no audio needed)

**Cons**:
- Requires training data or initial patterns
- HMM complexity can be high for long sequences
- May need pattern length constraints

---

### 3. Principal Component Analysis (PCA) - Eigenrhythms

**Project**: Eigenrhythms: Drum Pattern Basis Sets for Classification and Generation  
**URL**: https://www.ee.columbia.edu/~dpwe/pubs/ismir04-eigenrhythm.pdf

**Approach**:
- Normalizes and aligns drum patterns
- Applies PCA to extract basis patterns (eigenrhythms)
- Low-dimensional representation for classification
- Can approximate and interpolate patterns

**Key Techniques**:
- Pattern normalization (time alignment)
- PCA for dimensionality reduction
- Basis pattern extraction
- Pattern reconstruction from components

**Applicability to EchoZero**:
- ✅ Can work with EventData (convert to feature vectors)
- ✅ Good for pattern classification
- ✅ Can generate/reconstruct patterns
- ⚠️ Requires pattern alignment (temporal normalization)
- ⚠️ Needs multiple pattern examples for PCA

**Libraries/Implementation**:
- `scikit-learn.decomposition.PCA` (already in project)
- numpy for pattern vectorization
- Custom alignment algorithm (DTW or simple quantization)

**Pros**:
- Mathematical foundation
- Good for pattern classification
- Can generate new patterns
- Uses existing scikit-learn

**Cons**:
- Requires aligned patterns (temporal normalization challenge)
- Less suitable for online/incremental learning
- Pattern length must be fixed or quantized

---

### 4. Dynamic Time Warping (DTW) for Pattern Matching

**Project**: Various MIR research (common technique)

**Approach**:
- DTW finds optimal alignment between two sequences
- Handles tempo variations and timing differences
- Can match patterns with different lengths
- Used for query-by-example pattern search

**Key Techniques**:
- DTW algorithm for sequence alignment
- Distance metrics for pattern similarity
- Template matching with temporal flexibility
- Pattern clustering based on DTW distance

**Applicability to EchoZero**:
- ✅ Excellent for EventData (time sequences)
- ✅ Handles timing variations naturally
- ✅ Can find similar patterns in messy data
- ✅ Works with classification sequences
- ✅ Can be used for online pattern detection

**Libraries/Implementation**:
- `dtaidistance` (fast DTW implementation)
- `fastdtw` (approximate DTW, faster)
- `scipy.spatial.distance` (custom DTW)
- librosa has some DTW utilities

**Pros**:
- Handles timing variations excellently
- Well-established in MIR
- Good for finding similar patterns
- Can work incrementally

**Cons**:
- O(n*m) complexity (can be slow for long sequences)
- Requires pattern templates or examples
- May need constraints for real-time use

---

### 5. Online Learning with k-Nearest Neighbors (k-NN)

**Project**: User Specific Adaptation in Automatic Transcription of Vocalised Percussion  
**URL**: https://arxiv.org/abs/1811.02406

**Approach**:
- Live Vocalised Transcription (LVT) system
- User-specific training with k-NN classifier
- Onset detection + feature extraction + ML classification
- Adapts to individual users incrementally

**Key Techniques**:
- k-NN for pattern classification
- Feature extraction from audio/events
- Incremental learning (add examples as they come)
- User-specific adaptation

**Applicability to EchoZero**:
- ✅ Can learn patterns incrementally (online)
- ✅ Works with EventData features
- ✅ Simple to implement
- ✅ Adapts to user data
- ✅ Can use existing classification as features

**Libraries/Implementation**:
- `scikit-learn.neighbors.KNeighborsClassifier` (already in project)
- Custom feature extraction from EventData
- Incremental learning wrapper

**Pros**:
- Simple and interpretable
- Online learning friendly
- No training phase required
- Uses existing scikit-learn

**Cons**:
- Requires good feature representation
- k-NN can be slow with many examples
- May need feature engineering
- Less sophisticated than HMM/DTW

---

## Library Recommendations

### Already Available in EchoZero

1. **librosa** (0.10.1+)
   - Onset detection, tempo estimation
   - Audio feature extraction
   - Some DTW utilities

2. **scikit-learn** (1.3.0+)
   - PCA, k-NN, clustering
   - Feature extraction utilities
   - Distance metrics

3. **numpy/scipy** (1.26.0+, 1.14.0+)
   - Signal processing
   - Array operations
   - Distance calculations

4. **PyTorch/TensorFlow**
   - Deep learning for pattern recognition (if needed)
   - Sequence models (RNN, LSTM, Transformer)

### Recommended Additions

1. **dtaidistance** (for DTW)
   ```bash
   pip install dtaidistance
   ```
   - Fast DTW implementation
   - C-optimized
   - Good for pattern matching

2. **hmmlearn** (for HMM)
   ```bash
   pip install hmmlearn
   ```
   - HMM implementation
   - Sequence modeling
   - Pattern prediction

3. **music21** (optional, for music theory)
   ```bash
   pip install music21
   ```
   - Music analysis
   - Rhythm analysis
   - Pattern quantization

### Alternative: Custom Implementation

For lightweight solutions, can implement:
- Simple DTW (using numpy)
- Basic HMM (using numpy)
- Pattern matching (using difflib or custom)

---

## Hybrid Approach Recommendation

Based on the study, a **hybrid approach** combining multiple techniques would be most effective:

### Phase 1: Pattern Detection (DTW + Template Matching)
- Use DTW to find similar event sequences
- Match against learned pattern templates
- Handle timing variations

### Phase 2: Pattern Learning (HMM or k-NN)
- Learn patterns incrementally as events arrive
- Use HMM for sequence prediction
- Or k-NN for pattern classification

### Phase 3: Pattern Prediction
- Predict next events based on learned patterns
- Fill in missing events in patterns
- Handle noisy/messy data

### Phase 4: Similarity Detection (Audio Features)
- If AudioDataItem available, extract features
- Cluster similar-sounding events
- Use for pattern refinement

---

## Implementation Strategy

### Input Requirements

**Required**:
- `EventDataItem` with percussion events (time, classification)

**Optional** (for advanced features):
- `AudioDataItem` for audio similarity detection
- Pattern templates (user-provided or learned)

### Output

- `EventDataItem` with:
  - Original events
  - Detected patterns (pattern IDs, confidence)
  - Predicted events (filled-in pattern completions)
  - Pattern metadata (tempo, length, variations)

### Block Configuration

```python
{
    "pattern_length_min": 4,        # Minimum pattern length (events)
    "pattern_length_max": 32,        # Maximum pattern length
    "similarity_threshold": 0.7,     # DTW similarity threshold
    "prediction_enabled": True,      # Enable pattern prediction
    "learning_mode": "incremental",  # incremental | batch
    "method": "hybrid",              # dtw | hmm | pca | knn | hybrid
    "use_audio_similarity": False,   # Use audio features if available
}
```

### Processing Flow

1. **Event Collection**: Gather events from input EventDataItem
2. **Pattern Detection**: Find repeating patterns using DTW
3. **Pattern Learning**: Update pattern models (HMM/k-NN)
4. **Pattern Prediction**: Predict missing/completing events
5. **Output Generation**: Create EventDataItem with patterns + predictions

---

## Technical Considerations

### Temporal Normalization

Events may have varying tempos. Options:
- Quantize to grid (simple)
- Use DTW for alignment (flexible)
- Normalize by tempo (if detected)

### Online vs Batch Processing

- **Online**: Process events as they arrive (incremental)
- **Batch**: Process complete event sequence

Recommendation: Support both modes, default to online for real-time feel.

### Pattern Storage

- Store learned patterns in block metadata
- Or create separate PatternDataItem type
- Persist patterns across project saves

### Performance

- DTW can be slow for long sequences → use fastdtw or constraints
- HMM training can be expensive → use incremental updates
- k-NN needs efficient storage → use approximate nearest neighbors

---

## Next Steps

1. **Prototype**: Start with DTW-based pattern detection (simplest)
2. **Add Learning**: Implement incremental pattern learning (k-NN or HMM)
3. **Add Prediction**: Predict next events based on patterns
4. **Add Audio Similarity**: If AudioDataItem available, use for refinement
5. **Optimize**: Performance tuning for real-time use

---

## References

1. Masataka Goto - Sound Source Separation for Percussion Instruments
   - https://staff.aist.go.jp/m.goto/PROJ/sss.html

2. Tabla Pattern Discovery (HMM + RLCS)
   - https://compmusic.upf.edu/ismir-2015-tabla

3. Eigenrhythms (PCA-based)
   - https://www.ee.columbia.edu/~dpwe/pubs/ismir04-eigenrhythm.pdf

4. Voice Drummer (k-NN + user adaptation)
   - https://staff.aist.go.jp/t.nakano/VoiceDrummer/
   - https://arxiv.org/abs/1811.02406

5. Beijing Opera Percussion Pattern Dataset
   - https://compmusic.upf.edu/bopp-dataset

6. PANNs (Pretrained Audio Neural Networks)
   - https://arxiv.org/abs/1912.10211

---

## Conclusion

The most promising approach for EchoZero is a **hybrid system** combining:

1. **DTW** for flexible pattern matching (handles timing variations)
2. **HMM or k-NN** for incremental pattern learning
3. **Audio features** (if available) for similarity detection

This provides:
- Robust pattern detection in messy data
- Online/incremental learning
- Pattern prediction capabilities
- Flexibility to work with EventData alone or with audio

Start with DTW + k-NN (simplest), then add HMM for better sequence modeling if needed.

