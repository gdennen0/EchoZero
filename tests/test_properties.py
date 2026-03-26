"""
Property-Based + Round-Trip Serialization Tests for EchoZero 2.

SQLite principle: serialize → deserialize → serialize must be identical.
Also: for random inputs, invariants must always hold (manual parametrized,
no hypothesis dependency).

Covers:
- Round-trip: Take (EventData + AudioData), TakeLayer, Graph, full project
- Properties of merge_events: additive count, intersect is subset, etc.
- Properties of TakeLayer: snapshot/restore preserves invariants
- Properties of take ID generation: always unique
- Properties of ExecutionCache: store/get symmetry
- Properties of ready_nodes: never returns running nodes
"""

import json
import os
import random
import tempfile

import pytest

from datetime import datetime, timezone

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.cache import ExecutionCache
from echozero.coordinator import ready_nodes
from echozero.serialization import (
    deserialize_graph,
    deserialize_take,
    deserialize_take_layer,
    load_project,
    save_project,
    serialize_graph,
    serialize_take,
    serialize_take_layer,
)
from echozero.takes import (
    Take,
    TakeLayer,
    TakeLayerError,
    TakeSource,
    merge_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(time: float, duration: float = 0.1, extra_meta: dict = None) -> Event:
    return Event(
        id=f"e-{time}",
        time=time,
        duration=duration,
        classifications={"type": "onset", "confidence": 0.9},
        metadata=extra_meta or {"note": "A4"},
        origin="test",
    )


def _event_data(*times: float) -> EventData:
    if not times:
        return EventData(layers=())
    events = tuple(_evt(t) for t in times)
    layer = Layer(id="l1", name="Layer 1", events=events)
    return EventData(layers=(layer,))


def _main_take(label: str = "Main", *times: float) -> Take:
    if not times:
        times = (1.0, 2.0)
    return Take.create(data=_event_data(*times), label=label, is_main=True)


def _take(label: str = "Take", *times: float) -> Take:
    if not times:
        times = (1.0,)
    return Take.create(data=_event_data(*times), label=label, is_main=False)


def _make_block(bid: str, btype: str = "processor") -> Block:
    return Block(
        id=bid, name=bid, block_type=btype,
        category=BlockCategory.PROCESSOR,
        input_ports=(Port(name="in", port_type=PortType.EVENT, direction=Direction.INPUT),),
        output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
    )


def _make_root_block(bid: str) -> Block:
    return Block(
        id=bid, name=bid, block_type="source",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
    )


# ===========================================================================
# Round-trip: Take serialization
# ===========================================================================


class TestTakeRoundTrip:

    def test_take_with_event_data_round_trips(self):
        take = Take.create(
            data=_event_data(1.0, 2.0, 3.0),
            label="Test Take",
            origin="pipeline",
            source=TakeSource(
                block_id="b1", block_type="Detector",
                settings_snapshot={"threshold": 0.5, "window": 1024},
                run_id="r1",
            ),
            is_main=True,
            notes="First round-trip",
        )
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        s2 = serialize_take(d)
        assert s1 == s2

    def test_take_with_audio_data_round_trips(self):
        audio = AudioData(sample_rate=44100, duration=30.5, file_path="/audio/file.wav", channel_count=2)
        take = Take.create(data=audio, label="Audio Take", origin="sync", is_main=False)
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        s2 = serialize_take(d)
        assert s1 == s2

    def test_take_no_source_round_trips(self):
        take = Take.create(data=_event_data(1.0), label="User Edit", origin="user")
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        s2 = serialize_take(d)
        assert s1 == s2
        assert d.source is None

    def test_take_empty_event_data_round_trips(self):
        """EventData with no layers."""
        take = Take.create(data=EventData(layers=()), label="Empty", origin="pipeline", is_main=True)
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        s2 = serialize_take(d)
        assert s1 == s2

    def test_take_multi_layer_event_data_round_trips(self):
        """EventData with multiple layers."""
        l1 = Layer(id="l1", name="L1", events=(_evt(1.0), _evt(2.0)))
        l2 = Layer(id="l2", name="L2", events=(_evt(3.0),))
        take = Take.create(data=EventData(layers=(l1, l2)), label="Multi", origin="merge", is_main=True)
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        s2 = serialize_take(d)
        assert s1 == s2
        assert len(d.data.layers) == 2

    def test_take_with_nested_metadata_round_trips(self):
        """Event with deeply nested metadata/classifications."""
        evt = Event(
            id="e1", time=1.5, duration=0.2,
            classifications={"nested": {"a": [1, 2, {"b": True}]}},
            metadata={"tags": ["kick", "transient"], "confidence": 0.95},
            origin="ml",
        )
        layer = Layer(id="l1", name="L", events=(evt,))
        take = Take.create(data=EventData(layers=(layer,)), label="Complex", origin="pipeline", is_main=True)
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        s2 = serialize_take(d)
        assert s1 == s2

    def test_take_timestamps_preserved(self):
        """Datetime round-trips faithfully."""
        take = Take.create(data=_event_data(1.0), label="T", is_main=True)
        s1 = serialize_take(take)
        d = deserialize_take(s1)
        assert d.created_at == take.created_at

    def test_all_take_origins_round_trip(self):
        for origin in ("pipeline", "user", "merge", "sync"):
            take = Take.create(data=_event_data(1.0), label=origin, origin=origin, is_main=True)  # type: ignore
            s1 = serialize_take(take)
            d = deserialize_take(s1)
            s2 = serialize_take(d)
            assert s1 == s2, f"Round-trip failed for origin={origin}"


# ===========================================================================
# Round-trip: TakeLayer serialization
# ===========================================================================


class TestTakeLayerRoundTrip:

    def test_single_take_layer_round_trips(self):
        layer = TakeLayer(layer_id="l1", takes=[_main_take("M")])
        s1 = serialize_take_layer(layer)
        d = deserialize_take_layer(s1)
        s2 = serialize_take_layer(d)
        assert s1 == s2

    def test_multi_take_layer_round_trips(self):
        main = _main_take("Main", 1.0, 2.0)
        t1 = _take("Take 1", 3.0)
        t2 = _take("Take 2", 4.0, 5.0)
        layer = TakeLayer(layer_id="layer-1", takes=[main, t1, t2])
        s1 = serialize_take_layer(layer)
        d = deserialize_take_layer(s1)
        s2 = serialize_take_layer(d)
        assert s1 == s2
        assert d.take_count == 3

    def test_empty_layer_round_trips(self):
        layer = TakeLayer(layer_id="empty", takes=[])
        s1 = serialize_take_layer(layer)
        d = deserialize_take_layer(s1)
        s2 = serialize_take_layer(d)
        assert s1 == s2

    def test_main_flag_preserved_after_round_trip(self):
        main = _main_take("Main")
        non_main = _take("T1")
        layer = TakeLayer(layer_id="l", takes=[main, non_main])
        s1 = serialize_take_layer(layer)
        d = deserialize_take_layer(s1)
        assert d.main_take().id == main.id
        assert not d.takes[1].is_main

    def test_mixed_audio_and_event_layer_round_trips(self):
        audio = AudioData(sample_rate=48000, duration=10.0, file_path="audio.wav")
        event_take = _main_take("Events", 1.0)
        audio_take = Take.create(data=audio, label="Audio", origin="sync")
        layer = TakeLayer(layer_id="mixed", takes=[event_take, audio_take])
        s1 = serialize_take_layer(layer)
        d = deserialize_take_layer(s1)
        s2 = serialize_take_layer(d)
        assert s1 == s2


# ===========================================================================
# Round-trip: Graph serialization
# ===========================================================================


class TestGraphRoundTrip:

    def test_empty_graph_round_trips(self):
        g = Graph()
        s1 = serialize_graph(g)
        g2 = deserialize_graph(s1)
        s2 = serialize_graph(g2)
        assert s1 == s2

    def test_single_block_graph_round_trips(self):
        # deserialize_graph always loads blocks as STALE (documented behavior).
        # Round-trip property: serialize(deserialize(data)) is stable on the
        # second pass — i.e., s2 == s3 (fixed point).
        g = Graph()
        g.add_block(_make_root_block("b1"))
        s1 = serialize_graph(g)
        g2 = deserialize_graph(s1)   # blocks become STALE
        s2 = serialize_graph(g2)
        g3 = deserialize_graph(s2)
        s3 = serialize_graph(g3)
        assert s2 == s3, "Graph serialization is not idempotent after first load"
        assert "b1" in g2.blocks

    def test_graph_with_connections_round_trips(self):
        # Same fixed-point approach: s2 == s3
        g = Graph()
        g.add_block(_make_root_block("root"))
        g.add_block(_make_block("mid"))
        g.add_block(_make_block("leaf"))
        g.add_connection(Connection(
            source_block_id="root", source_output_name="out",
            target_block_id="mid", target_input_name="in",
        ))
        g.add_connection(Connection(
            source_block_id="mid", source_output_name="out",
            target_block_id="leaf", target_input_name="in",
        ))
        s1 = serialize_graph(g)
        g2 = deserialize_graph(s1)
        s2 = serialize_graph(g2)
        g3 = deserialize_graph(s2)
        s3 = serialize_graph(g3)
        assert s2 == s3
        assert len(g2.connections) == 2

    def test_all_block_categories_round_trip(self):
        g = Graph()
        for i, cat in enumerate(BlockCategory):
            b = Block(
                id=f"b{i}", name=f"Block-{cat.name}", block_type="x",
                category=cat,
                input_ports=(),
                output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
            )
            g.add_block(b)
        s1 = serialize_graph(g)
        g2 = deserialize_graph(s1)
        s2 = serialize_graph(g2)
        g3 = deserialize_graph(s2)
        s3 = serialize_graph(g3)
        assert s2 == s3

    def test_block_with_settings_round_trips(self):
        b = Block(
            id="b1", name="Settings Block", block_type="proc",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
            settings=BlockSettings(entries={"threshold": 0.5, "window": 1024, "mode": "fast"}),
        )
        g = Graph()
        g.add_block(b)
        s1 = serialize_graph(g)
        g2 = deserialize_graph(s1)
        assert g2.blocks["b1"].settings.entries == b.settings.entries

    def test_deserialized_graph_blocks_are_stale(self):
        """Deserialized graph always loads blocks as STALE (documented behavior)."""
        g = Graph()
        root = _make_root_block("r")
        g.add_block(root)
        # Mark as FRESH
        g.set_block_state("r", BlockState.FRESH)
        s1 = serialize_graph(g)
        g2 = deserialize_graph(s1)
        assert g2.blocks["r"].state == BlockState.STALE


# ===========================================================================
# Round-trip: Full project save/load
# ===========================================================================


class TestProjectRoundTrip:

    def test_save_load_empty_project(self, tmp_path):
        path = str(tmp_path / "project.json")
        g = Graph()
        save_project(path, g, take_layers=[])
        g2, layers = load_project(path)
        assert len(g2.blocks) == 0
        assert len(layers) == 0

    def test_save_load_graph_with_takes(self, tmp_path):
        path = str(tmp_path / "project.json")
        g = Graph()
        g.add_block(_make_root_block("source"))
        g.add_block(_make_block("proc"))
        g.add_connection(Connection(
            source_block_id="source", source_output_name="out",
            target_block_id="proc", target_input_name="in",
        ))
        main = _main_take("Main", 1.0, 2.0)
        t1 = _take("T1", 3.0)
        layer = TakeLayer(layer_id="l1", takes=[main, t1])

        save_project(path, g, take_layers=[layer])
        g2, layers2 = load_project(path)

        assert "source" in g2.blocks
        assert "proc" in g2.blocks
        assert len(g2.connections) == 1
        assert len(layers2) == 1
        assert layers2[0].take_count == 2
        assert layers2[0].main_take().label == "Main"

    def test_saved_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "project.json")
        g = Graph()
        g.add_block(_make_root_block("r"))
        save_project(path, g)
        with open(path) as f:
            data = json.load(f)
        assert "graph" in data
        assert "version" in data

    def test_save_load_multiple_take_layers(self, tmp_path):
        path = str(tmp_path / "project.json")
        g = Graph()
        layers = [
            TakeLayer(layer_id=f"l{i}", takes=[_main_take(f"Main-{i}")])
            for i in range(5)
        ]
        save_project(path, g, take_layers=layers)
        _, loaded = load_project(path)
        assert len(loaded) == 5
        for i, layer in enumerate(loaded):
            assert layer.layer_id == f"l{i}"


# ===========================================================================
# Property: merge_events invariants (manual parametrized)
# ===========================================================================


class TestMergeEventsProperties:

    @pytest.mark.parametrize("seed", range(15))
    def test_additive_count_is_sum(self, seed):
        """|additive(A, B)| == |A| + |B| — always."""
        rng = random.Random(seed)
        n_t = rng.randint(0, 20)
        n_s = rng.randint(0, 20)
        target = tuple(_evt(rng.uniform(0, 100)) for _ in range(n_t))
        source = tuple(_evt(rng.uniform(0, 100)) for _ in range(n_s))
        result = merge_events(target, source, strategy="additive")
        assert len(result) == n_t + n_s

    @pytest.mark.parametrize("seed", range(15))
    def test_intersect_result_is_subset_of_target(self, seed):
        """intersect(A, B) ⊆ A — all result events come from target."""
        rng = random.Random(seed)
        target_times = [float(rng.randint(0, 50)) for _ in range(rng.randint(0, 15))]
        source_times = [float(rng.randint(0, 50)) for _ in range(rng.randint(0, 15))]
        target = tuple(_evt(t) for t in target_times)
        source = tuple(_evt(t) for t in source_times)
        result = merge_events(target, source, strategy="intersect", time_epsilon=0.0)
        target_time_set = {e.time for e in target}
        for e in result:
            assert e.time in target_time_set, f"{e.time} not in target"

    @pytest.mark.parametrize("seed", range(15))
    def test_subtract_result_is_subset_of_target(self, seed):
        """subtract(A, B) ⊆ A — subtraction never adds events."""
        rng = random.Random(seed)
        target = tuple(_evt(float(i)) for i in range(rng.randint(0, 20)))
        source = tuple(_evt(float(i)) for i in range(rng.randint(0, 20)))
        result = merge_events(target, source, strategy="subtract", time_epsilon=0.0)
        target_time_set = {e.time for e in target}
        for e in result:
            assert e.time in target_time_set

    @pytest.mark.parametrize("seed", range(10))
    def test_intersect_idempotent_with_self(self, seed):
        """intersect(A, A) == A — intersecting with yourself returns all."""
        rng = random.Random(seed)
        times = [float(rng.randint(0, 100)) for _ in range(rng.randint(1, 20))]
        target = tuple(_evt(t) for t in times)
        result = merge_events(target, target, strategy="intersect", time_epsilon=0.0)
        assert len(result) == len(target)

    @pytest.mark.parametrize("seed", range(10))
    def test_subtract_with_self_leaves_nothing(self, seed):
        """subtract(A, A) with epsilon=0 removes all matched events."""
        rng = random.Random(seed)
        # Use integer times to guarantee exact float equality
        times = [float(rng.randint(0, 100)) for _ in range(rng.randint(0, 15))]
        target = tuple(_evt(t) for t in times)
        result = merge_events(target, target, strategy="subtract", time_epsilon=0.0)
        assert len(result) == 0

    @pytest.mark.parametrize("seed", range(10))
    def test_replace_range_preserves_sort_order(self, seed):
        """replace_range always returns events sorted by time."""
        rng = random.Random(seed)
        target = tuple(_evt(float(rng.randint(0, 100))) for _ in range(rng.randint(0, 10)))
        source = tuple(_evt(float(rng.randint(20, 80))) for _ in range(rng.randint(0, 10)))
        result = merge_events(
            target, source,
            strategy="replace_range",
            time_range=(20.0, 80.0),
        )
        times = [e.time for e in result]
        assert times == sorted(times), f"Not sorted: {times}"


# ===========================================================================
# Property: TakeLayer snapshot/restore
# ===========================================================================


class TestSnapshotRestoreProperties:

    @pytest.mark.parametrize("seed", range(10))
    def test_snapshot_restore_preserves_take_count(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 10)
        main = _main_take("Main")
        others = [_take(f"T{i}") for i in range(n - 1)]
        layer = TakeLayer(layer_id="l", takes=[main] + others)
        snap = layer.snapshot()

        # Apply random mutations
        for _ in range(rng.randint(0, 5)):
            new_take = _take(f"Added-{rng.randint(0, 999)}")
            layer.add_take(new_take)

        layer.restore(snap)
        assert layer.take_count == n

    @pytest.mark.parametrize("seed", range(10))
    def test_snapshot_restore_preserves_main_id(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 8)
        main = _main_take("Main")
        others = [_take(f"T{i}") for i in range(n - 1)]
        layer = TakeLayer(layer_id="l", takes=[main] + others)
        original_main_id = layer.main_take().id
        snap = layer.snapshot()

        # Promote a random non-main take
        non_mains = [t for t in layer.takes if not t.is_main]
        if non_mains:
            layer.promote_to_main(rng.choice(non_mains).id)

        layer.restore(snap)
        assert layer.main_take().id == original_main_id

    def test_snapshot_is_frozen(self):
        """TakeLayerSnapshot is a frozen dataclass (immutable)."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        snap = layer.snapshot()
        with pytest.raises(AttributeError):
            snap.layer_id = "mutated"  # type: ignore


# ===========================================================================
# Property: Take ID uniqueness
# ===========================================================================


class TestTakeIdUniqueness:

    @pytest.mark.parametrize("n", [1, 10, 100, 1000])
    def test_n_takes_have_n_unique_ids(self, n):
        ids = {Take.create(data=_event_data(1.0), label=f"T{i}").id for i in range(n)}
        assert len(ids) == n, f"Collision in {n} takes"


# ===========================================================================
# Property: ExecutionCache store/get symmetry
# ===========================================================================


class TestCacheStoreGetSymmetry:

    @pytest.mark.parametrize("seed", range(10))
    def test_stored_values_retrievable(self, seed):
        rng = random.Random(seed)
        cache = ExecutionCache()
        stored = {}
        for i in range(rng.randint(1, 20)):
            block_id = f"block-{rng.randint(0, 5)}"
            port = rng.choice(["out", "events", "audio"])
            value = rng.random()
            cache.store(block_id, port, value, f"exec-{i}")
            stored[(block_id, port)] = value

        for (bid, port), expected in stored.items():
            entry = cache.get(bid, port)
            assert entry is not None
            assert entry.value == expected

    @pytest.mark.parametrize("seed", range(5))
    def test_invalidate_removes_only_target_block(self, seed):
        rng = random.Random(seed)
        cache = ExecutionCache()
        block_ids = [f"b{i}" for i in range(5)]
        for bid in block_ids:
            cache.store(bid, "out", rng.random(), "exec-1")

        target = rng.choice(block_ids)
        cache.invalidate(target)

        assert not cache.has_valid_output(target, "out")
        for bid in block_ids:
            if bid != target:
                assert cache.has_valid_output(bid, "out"), f"{bid} should still be cached"


# ===========================================================================
# Property: ready_nodes never returns running nodes
# ===========================================================================


class TestReadyNodesProperty:

    def _make_independent_graph(self, n: int) -> Graph:
        """n independent root blocks (no connections)."""
        g = Graph()
        for i in range(n):
            g.add_block(_make_root_block(f"b{i}"))
        return g

    @pytest.mark.parametrize("seed", range(10))
    def test_ready_never_includes_running(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 10)
        g = self._make_independent_graph(n)
        cache = ExecutionCache()

        all_ids = {f"b{i}" for i in range(n)}
        dirty = {bid for bid in all_ids if rng.random() > 0.3}
        running = {bid for bid in dirty if rng.random() > 0.5}

        result = ready_nodes(g, dirty=dirty, running=running, cache=cache)
        assert result.isdisjoint(running), f"Running nodes in ready: {result & running}"

    @pytest.mark.parametrize("seed", range(10))
    def test_ready_is_subset_of_dirty(self, seed):
        rng = random.Random(seed)
        n = rng.randint(2, 10)
        g = self._make_independent_graph(n)
        cache = ExecutionCache()

        all_ids = {f"b{i}" for i in range(n)}
        dirty = {bid for bid in all_ids if rng.random() > 0.4}
        running = set()

        result = ready_nodes(g, dirty=dirty, running=running, cache=cache)
        assert result.issubset(dirty), f"Ready contains non-dirty nodes: {result - dirty}"
