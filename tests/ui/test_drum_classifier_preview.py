from echozero.ui.qt.timeline.drum_classifier_preview import label_drum_hit


def test_label_drum_hit_kick_branch():
    label, conf = label_drum_hit(low_ratio=0.45, centroid_hz=1200.0, zcr=0.08, rms=0.04)
    assert label == "kick"
    assert conf > 0.55


def test_label_drum_hit_hihat_branch():
    label, conf = label_drum_hit(low_ratio=0.05, centroid_hz=7000.0, zcr=0.28, rms=0.02)
    assert label == "hihat"
    assert conf > 0.5


def test_label_drum_hit_snare_branch():
    label, conf = label_drum_hit(low_ratio=0.12, centroid_hz=3200.0, zcr=0.12, rms=0.05)
    assert label == "snare"
    assert conf > 0.5


def test_label_drum_hit_fallback_clap():
    label, conf = label_drum_hit(low_ratio=0.08, centroid_hz=1800.0, zcr=0.10, rms=0.01)
    assert label == "clap"
    assert conf == 0.45
