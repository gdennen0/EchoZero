"""Song title extraction tests for import filename cleanup.
Exists to keep the import title heuristics stable for common timecode naming patterns.
Connects song import preferences to deterministic filename-to-title expectations.
"""

from echozero.application.settings.models import SongImportNameMode
from echozero.application.song.title_extraction import resolve_import_song_titles


def test_resolve_import_song_titles_keeps_filename_by_default() -> None:
    resolved = resolve_import_song_titles(
        ("/tmp/NoahKahan_PaidTimeOff_87bpm_SMPTE_v01.wav",),
        name_mode=SongImportNameMode.FILENAME,
    )

    assert resolved["/tmp/NoahKahan_PaidTimeOff_87bpm_SMPTE_v01.wav"] == (
        "NoahKahan_PaidTimeOff_87bpm_SMPTE_v01"
    )


def test_resolve_import_song_titles_extracts_clean_title_from_single_reference_name() -> None:
    resolved = resolve_import_song_titles(
        ("/tmp/NoahKahan_PaidTimeOff_87bpm_SMPTE_v01.wav",),
        name_mode=SongImportNameMode.EXTRACT_TITLE,
    )

    assert resolved["/tmp/NoahKahan_PaidTimeOff_87bpm_SMPTE_v01.wav"] == "Paid Time Off"


def test_resolve_import_song_titles_removes_shared_artist_prefix_across_batch() -> None:
    audio_paths = (
        "/tmp/NoahKahan_PaidTimeOff_87bpm_SMPTE_v01.wav",
        "/tmp/NoahKahan_StickSeason_115bpm_SMPTE_v01.wav",
        "/tmp/NoahKahan_YoureGonnaGoFar_85bpm_SMPTE_v01.wav",
    )

    resolved = resolve_import_song_titles(
        audio_paths,
        name_mode=SongImportNameMode.EXTRACT_TITLE,
    )

    assert resolved[audio_paths[0]] == "Paid Time Off"
    assert resolved[audio_paths[1]] == "Stick Season"
    assert resolved[audio_paths[2]] == "Youre Gonna Go Far"


def test_resolve_import_song_titles_preserves_non_artist_two_word_titles() -> None:
    resolved = resolve_import_song_titles(
        ("/tmp/Set_Intro_143bpm_SMPTE_v01.wav",),
        name_mode=SongImportNameMode.EXTRACT_TITLE,
    )

    assert resolved["/tmp/Set_Intro_143bpm_SMPTE_v01.wav"] == "Set Intro"
