"""
Tests for the Take System.

Tests the five invariants:
1. Every TakeLayer has exactly one Take where is_main=True.
2. A Take's data is never mutated after creation.
3. Take IDs are globally unique and never reused.
4. Sync only reads/writes the Take where is_main=True.
5. Undo always restores a previous valid TakeLayer state.
"""

import pytest
from datetime import datetime, timezone

from echozero.domain.types import Event, EventData, Layer, AudioData
from echozero.takes import (
    Take,
    TakeLayer,
    TakeLayerError,
    TakeLayerSnapshot,
    TakeSource,
    merge_events,
    merge_take_into,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event(time: float, label: str = "test") -> Event:
    return Event(
        id=f"evt-{time}",
        time=time,
        duration=0.1,
        classifications={},
        metadata={},
        origin="test",
    )


def _make_event_data(*times: float) -> EventData:
    events = tuple(_make_event(t) for t in times)
    layer = Layer(id="layer-1", name="Test Layer", events=events)
    return EventData(layers=(layer,))


def _make_main_take(label: str = "Main", times: tuple = (1.0, 2.0, 3.0)) -> Take:
    return Take.create(
        data=_make_event_data(*times),
        label=label,
        origin="user",
        is_main=True,
    )


def _make_take(label: str = "Take", times: tuple = (1.0, 2.0, 3.0)) -> Take:
    return Take.create(
        data=_make_event_data(*times),
        label=label,
        origin="pipeline",
        source=TakeSource(
            block_id="block-1",
            block_type="DetectOnsets",
            settings_snapshot={"threshold": 0.5},
            run_id="run-1",
        ),
    )


def _make_layer(main_times=(1.0, 2.0, 3.0), extra_takes=0) -> TakeLayer:
    main = _make_main_take(times=main_times)
    takes = [main]
    for i in range(extra_takes):
        takes.append(_make_take(label=f"Take {i + 1}", times=(1.1 * (i + 1),)))
    return TakeLayer(layer_id="test-layer", takes=takes)


# ---------------------------------------------------------------------------
# Invariant 1: Exactly one main take
# ---------------------------------------------------------------------------


class TestMainInvariant:
    def test_layer_requires_exactly_one_main(self):
        main = _make_main_take()
        layer = TakeLayer(layer_id="test", takes=[main])
        assert layer.main_take().id == main.id

    def test_layer_rejects_zero_main(self):
        take = _make_take()  # is_main=False
        with pytest.raises(TakeLayerError, match="0 main takes"):
            TakeLayer(layer_id="test", takes=[take])

    def test_layer_rejects_two_mains(self):
        m1 = _make_main_take(label="Main 1")
        m2 = _make_main_take(label="Main 2")
        with pytest.raises(TakeLayerError, match="2 main takes"):
            TakeLayer(layer_id="test", takes=[m1, m2])

    def test_empty_layer_is_valid(self):
        layer = TakeLayer(layer_id="test", takes=[])
        assert layer.take_count == 0

    def test_promote_swaps_main(self):
        layer = _make_layer(extra_takes=1)
        old_main = layer.main_take()
        new_take = layer.takes[1]
        layer.promote_to_main(new_take.id)
        assert layer.main_take().id == new_take.id
        # Old main is no longer main
        old = layer.get_take(old_main.id)
        assert not old.is_main


# ---------------------------------------------------------------------------
# Invariant 2: Takes are immutable (frozen dataclass)
# ---------------------------------------------------------------------------


class TestTakeImmutability:
    def test_take_is_frozen(self):
        take = _make_take()
        with pytest.raises(AttributeError):
            take.label = "mutated"  # type: ignore

    def test_take_data_is_frozen(self):
        take = _make_take()
        with pytest.raises(AttributeError):
            take.data = _make_event_data(9.0)  # type: ignore


# ---------------------------------------------------------------------------
# Invariant 3: Unique IDs
# ---------------------------------------------------------------------------


class TestUniqueIds:
    def test_factory_produces_unique_ids(self):
        t1 = _make_take()
        t2 = _make_take()
        assert t1.id != t2.id

    def test_hundred_takes_all_unique(self):
        ids = {Take.create(data=_make_event_data(1.0), label=f"T{i}").id for i in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Invariant 4: Sync only sees main
# ---------------------------------------------------------------------------


class TestSyncBoundary:
    def test_main_take_returns_main_only(self):
        layer = _make_layer(extra_takes=3)
        main = layer.main_take()
        assert main.is_main
        # Other takes are not main
        for t in layer.takes:
            if t.id != main.id:
                assert not t.is_main


# ---------------------------------------------------------------------------
# Invariant 5: Undo via snapshots
# ---------------------------------------------------------------------------


class TestUndoSnapshots:
    def test_snapshot_captures_state(self):
        layer = _make_layer(extra_takes=2)
        snap = layer.snapshot()
        assert len(snap.takes) == 3
        assert snap.layer_id == "test-layer"

    def test_restore_from_snapshot(self):
        layer = _make_layer(extra_takes=2)
        snap = layer.snapshot()

        # Mutate: add a take
        layer.add_take(_make_take(label="New"))
        assert layer.take_count == 4

        # Restore
        layer.restore(snap)
        assert layer.take_count == 3

    def test_restore_preserves_main(self):
        layer = _make_layer(extra_takes=1)
        snap = layer.snapshot()
        original_main_id = layer.main_take().id

        # Promote the other take
        other = layer.takes[1]
        layer.promote_to_main(other.id)
        assert layer.main_take().id == other.id

        # Restore
        layer.restore(snap)
        assert layer.main_take().id == original_main_id


# ---------------------------------------------------------------------------
# Take lifecycle
# ---------------------------------------------------------------------------


class TestTakeLifecycle:
    def test_new_takes_are_not_main(self):
        take = _make_take()
        assert not take.is_main

    def test_add_take_rejects_main(self):
        layer = _make_layer()
        main_take = _make_main_take(label="Another Main")
        with pytest.raises(TakeLayerError, match="must not be main"):
            layer.add_take(main_take)

    def test_add_take_appends(self):
        layer = _make_layer()
        new_take = _make_take(label="New")
        layer.add_take(new_take)
        assert layer.take_count == 2
        assert layer.takes[-1].id == new_take.id

    def test_remove_take(self):
        layer = _make_layer(extra_takes=1)
        non_main = layer.takes[1]
        removed = layer.remove_take(non_main.id)
        assert removed.id == non_main.id
        assert layer.take_count == 1

    def test_cannot_remove_main(self):
        layer = _make_layer()
        main = layer.main_take()
        with pytest.raises(TakeLayerError, match="Cannot remove the main take"):
            layer.remove_take(main.id)

    def test_replace_take_in_place(self):
        layer = _make_layer(extra_takes=1)
        old_take = layer.takes[1]
        new_data = _make_event_data(9.0, 10.0)
        new_take = Take(
            id=old_take.id,
            label="Edited",
            data=new_data,
            origin="user",
            source=old_take.source,
            created_at=old_take.created_at,
            is_main=False,
        )
        layer.replace_take(old_take.id, new_take)
        assert layer.get_take(old_take.id).label == "Edited"

    def test_get_take_not_found(self):
        layer = _make_layer()
        with pytest.raises(TakeLayerError, match="not found"):
            layer.get_take("nonexistent-id")

    def test_reorder_takes(self):
        layer = _make_layer(extra_takes=2)
        ids = [t.id for t in layer.takes]
        reversed_ids = list(reversed(ids))
        layer.reorder_takes(reversed_ids)
        assert [t.id for t in layer.takes] == reversed_ids


# ---------------------------------------------------------------------------
# Take factory
# ---------------------------------------------------------------------------


class TestTakeFactory:
    def test_create_with_source(self):
        source = TakeSource(
            block_id="b1",
            block_type="DetectOnsets",
            settings_snapshot={"threshold": 0.3},
            run_id="r1",
        )
        take = Take.create(
            data=_make_event_data(1.0, 2.0),
            label="Test",
            origin="pipeline",
            source=source,
        )
        assert take.source is not None
        assert take.source.block_type == "DetectOnsets"
        assert take.origin == "pipeline"
        assert not take.is_main

    def test_create_user_authored(self):
        take = Take.create(
            data=_make_event_data(3.2),
            label="Manual Events",
            origin="user",
        )
        assert take.source is None
        assert take.origin == "user"

    def test_create_has_timestamp(self):
        before = datetime.now(timezone.utc)
        take = Take.create(data=_make_event_data(1.0), label="T")
        after = datetime.now(timezone.utc)
        assert before <= take.created_at <= after


# ---------------------------------------------------------------------------
# Merge: event-level strategies
# ---------------------------------------------------------------------------


class TestMergeEvents:
    def test_additive_unions_all(self):
        target = (_make_event(1.0), _make_event(2.0))
        source = (_make_event(3.0), _make_event(4.0))
        result = merge_events(target, source, strategy="additive")
        assert len(result) == 4

    def test_additive_keeps_duplicates(self):
        target = (_make_event(1.0),)
        source = (_make_event(1.0),)
        result = merge_events(target, source, strategy="additive")
        assert len(result) == 2  # Both kept

    def test_subtract_removes_matches(self):
        target = (_make_event(1.0), _make_event(2.0), _make_event(3.0))
        source = (_make_event(1.0), _make_event(3.0))
        result = merge_events(target, source, strategy="subtract")
        assert len(result) == 1
        assert result[0].time == 2.0

    def test_subtract_respects_epsilon(self):
        target = (_make_event(1.0), _make_event(2.0))
        source = (_make_event(1.03),)  # within default 0.05 epsilon
        result = merge_events(target, source, strategy="subtract")
        assert len(result) == 1
        assert result[0].time == 2.0

    def test_intersect_keeps_matches_only(self):
        target = (_make_event(1.0), _make_event(2.0), _make_event(3.0))
        source = (_make_event(2.0), _make_event(4.0))
        result = merge_events(target, source, strategy="intersect")
        assert len(result) == 1
        assert result[0].time == 2.0

    def test_replace_range(self):
        target = (_make_event(1.0), _make_event(2.0), _make_event(3.0), _make_event(4.0))
        source = (_make_event(2.5), _make_event(2.8))
        result = merge_events(
            target, source,
            strategy="replace_range",
            time_range=(2.0, 3.0),
        )
        # Events at 2.0 and 3.0 removed, replaced by 2.5 and 2.8
        times = [e.time for e in result]
        assert 1.0 in times
        assert 4.0 in times
        assert 2.5 in times
        assert 2.8 in times
        assert 2.0 not in times
        assert 3.0 not in times

    def test_replace_range_requires_range(self):
        with pytest.raises(ValueError, match="requires time_range"):
            merge_events((), (), strategy="replace_range")

    def test_replace_range_preserves_sort_order(self):
        target = (_make_event(1.0), _make_event(5.0))
        source = (_make_event(3.0), _make_event(2.0))
        result = merge_events(
            target, source,
            strategy="replace_range",
            time_range=(1.5, 4.0),
        )
        times = [e.time for e in result]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# Merge: take-level
# ---------------------------------------------------------------------------


class TestMergeTakes:
    def test_merge_take_into_main(self):
        layer = _make_layer(main_times=(1.0, 2.0), extra_takes=0)
        source = _make_take(label="Source", times=(3.0, 4.0))
        layer.add_take(source)

        merged = merge_take_into(layer, source.id, layer.main_take().id)
        assert isinstance(merged.data, EventData)
        events = merged.data.layers[0].events
        assert len(events) == 4  # 2 from main + 2 from source

    def test_merge_cherry_pick(self):
        layer = _make_layer(main_times=(1.0,), extra_takes=0)
        source = _make_take(label="Source", times=(2.0, 3.0, 4.0))
        layer.add_take(source)

        merged = merge_take_into(
            layer, source.id, layer.main_take().id,
            event_indices={0, 2},  # pick events at index 0 and 2
        )
        events = merged.data.layers[0].events
        assert len(events) == 3  # 1 from main + 2 cherry-picked

    def test_merge_preserves_main_flag(self):
        layer = _make_layer(main_times=(1.0,), extra_takes=0)
        source = _make_take(label="Source", times=(2.0,))
        layer.add_take(source)

        merged = merge_take_into(layer, source.id, layer.main_take().id)
        assert merged.is_main  # Main flag preserved

    def test_merge_sets_origin(self):
        layer = _make_layer(main_times=(1.0,), extra_takes=0)
        source = _make_take(label="Source", times=(2.0,))
        layer.add_take(source)

        merged = merge_take_into(layer, source.id, layer.main_take().id)
        assert merged.origin == "merge"

    def test_merge_with_subtract(self):
        layer = _make_layer(main_times=(1.0, 2.0, 3.0), extra_takes=0)
        source = _make_take(label="Remove These", times=(1.0, 3.0))
        layer.add_take(source)

        merged = merge_take_into(
            layer, source.id, layer.main_take().id,
            strategy="subtract",
        )
        events = merged.data.layers[0].events
        assert len(events) == 1
        assert events[0].time == 2.0

    def test_merge_audio_data_raises(self):
        audio = AudioData(sample_rate=44100, duration=3.0, file_path="test.wav")
        main = Take.create(data=audio, label="Audio", origin="pipeline", is_main=True)
        source = Take.create(data=audio, label="Audio 2", origin="pipeline")
        layer = TakeLayer(layer_id="audio-layer", takes=[main])
        layer.add_take(source)

        with pytest.raises(TakeLayerError, match="only supported for EventData"):
            merge_take_into(layer, source.id, main.id)
