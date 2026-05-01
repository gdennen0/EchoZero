from echozero.application.shared.cue_numbers import parse_positive_cue_number


def test_parse_positive_cue_number_accepts_fractional_values_below_one() -> None:
    assert parse_positive_cue_number(0.5) == 0.5
    assert parse_positive_cue_number("0.5") == 0.5


def test_parse_positive_cue_number_rejects_zero_or_negative_values() -> None:
    assert parse_positive_cue_number(0) is None
    assert parse_positive_cue_number("0") is None
    assert parse_positive_cue_number(-1) is None
