from __future__ import annotations


REQUIRED_PREPROCESSING_KEYS = {
    "sampleRate",
    "maxLength",
    "nFft",
    "hopLength",
    "nMels",
    "fmax",
}


SUPPORTED_CLASSIFICATION_MODES = {
    "multiclass",
    "binary",
    "positive_vs_other",
}
