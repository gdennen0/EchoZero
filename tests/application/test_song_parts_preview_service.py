from echozero.application.timeline.song_parts_preview_service import (
    build_song_parts_preview,
)
from echozero.processors.song_structure_mir import SongStructureAnalysis, SongStructureSegment


def test_build_song_parts_preview_uses_mir_analysis_projection(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"fixture")

    def _fake_analyze_song_structure(**_kwargs):
        return SongStructureAnalysis(
            duration_seconds=12.0,
            beat_times_seconds=(0.0, 4.0, 8.0),
            boundaries_seconds=(0.0, 4.0, 8.0),
            novelty_curve=(0.1, 0.8, 0.4),
            self_similarity_matrix=(
                (1.0, 0.3, 0.9),
                (0.3, 1.0, 0.2),
                (0.9, 0.2, 1.0),
            ),
            chroma_matrix=((0.0, 1.0, 0.0),),
            mel_spectrogram_db=((0.0, 0.0, 0.0),),
            embedding_points_2d=((0.0, 0.1), (0.7, 0.4), (-0.6, -0.3)),
            segments=(
                SongStructureSegment(0.0, "intro_01", "Intro", 0.9),
                SongStructureSegment(4.0, "verse_02", "Verse", 0.85),
                SongStructureSegment(8.0, "chorus_03", "Chorus", 0.88),
            ),
        )

    monkeypatch.setattr(
        "echozero.application.timeline.song_parts_preview_service.analyze_song_structure",
        _fake_analyze_song_structure,
    )

    preview = build_song_parts_preview(
        source_audio_path=str(audio_path),
        settings={"detect_method": "mir_self_similarity"},
    )

    assert preview.detect_method_label == "MIR Self-Similarity"
    assert len(preview.points) == 3
    assert preview.points[0].is_boundary is True
    assert preview.points[1].segment_index == 1
    assert preview.segments[2].label == "Chorus"
    assert "Vector-space preview combines" in preview.summary_text
