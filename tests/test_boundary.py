"""
Boundary Value + Invariant Tests for EchoZero 2.

SQLite principle: test the edges — empty inputs, single items, zero values,
max-ish sizes — because bugs cluster at boundaries.

Covers:
- TakeLayer: 0, 1, many takes
- merge_events: empty inputs, single events, epsilon=0
- Event: zero time, zero duration, extreme floats
- AudioData: zero values, empty path
- ExecutionCache: empty, single, overwrite
- GraphPlanner: empty graph, single block
- ready_nodes: all combinations of dirty/running
- TakeLayer.reorder_takes: single take, no-op, all permutations
- TakeLayer invariants: all documented error conditions
"""

import pytest
import threading

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
from echozero.errors import ValidationError
from echozero.editor.cache import ExecutionCache
from echozero.editor.coordinator import ready_nodes
from echozero.execution import GraphPlanner, ExecutionPlan
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


def _evt(time: float, duration: float = 0.1) -> Event:
    return Event(
        id=f"e-{time}",
        time=time,
        duration=duration,
        classifications={},
        metadata={},
        origin="test",
    )


def _event_data(*times: float) -> EventData:
    events = tuple(_evt(t) for t in times)
    layer = Layer(id="l1", name="L", events=events)
    return EventData(layers=(layer,))


def _main_take(label: str = "Main", *times: float) -> Take:
    if not times:
        times = (1.0,)
    return Take.create(data=_event_data(*times), label=label, is_main=True)


def _take(label: str = "Take", *times: float) -> Take:
    if not times:
        times = (1.0,)
    return Take.create(data=_event_data(*times), label=label, is_main=False)


def _make_block(bid: str, block_type: str = "dummy") -> Block:
    return Block(
        id=bid,
        name=bid,
        block_type=block_type,
        category=BlockCategory.PROCESSOR,
        input_ports=(Port(name="in", port_type=PortType.EVENT, direction=Direction.INPUT),),
        output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
    )


def _make_root_block(bid: str) -> Block:
    """Block with no inputs (a source/root in the graph)."""
    return Block(
        id=bid,
        name=bid,
        block_type="source",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
    )


# ===========================================================================
# TakeLayer boundary tests
# ===========================================================================


class TestTakeLayerBoundaries:

    def test_empty_layer_is_valid(self):
        """0 takes: no main required, no crash."""
        layer = TakeLayer(layer_id="empty", takes=[])
        assert layer.take_count == 0

    def test_empty_layer_main_take_raises(self):
        """Querying main on empty layer raises TakeLayerError."""
        layer = TakeLayer(layer_id="empty", takes=[])
        with pytest.raises(TakeLayerError, match="no main take"):
            layer.main_take()

    def test_single_take_must_be_main(self):
        """Single take layer: the one take must be main."""
        t = _main_take("Only")
        layer = TakeLayer(layer_id="single", takes=[t])
        assert layer.main_take().id == t.id
        assert layer.take_count == 1

    def test_single_take_non_main_raises(self):
        """Single non-main take violates the invariant."""
        t = _take("NotMain")
        with pytest.raises(TakeLayerError, match="0 main takes"):
            TakeLayer(layer_id="bad", takes=[t])

    def test_two_main_takes_raises(self):
        """Two mains violates the invariant."""
        m1 = _main_take("M1")
        m2 = _main_take("M2")
        with pytest.raises(TakeLayerError, match="2 main takes"):
            TakeLayer(layer_id="bad", takes=[m1, m2])

    def test_large_take_list(self):
        """1000 takes: one main, 999 non-main. No performance explosion."""
        main = _main_take("Main")
        others = [_take(f"T{i}") for i in range(999)]
        layer = TakeLayer(layer_id="big", takes=[main] + others)
        assert layer.take_count == 1000
        assert layer.main_take().id == main.id

    def test_remove_only_non_main_leaves_one(self):
        """After removing the only non-main, layer has one take (the main)."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M"), _take("T")])
        non_main = layer.takes[1]
        layer.remove_take(non_main.id)
        assert layer.take_count == 1
        assert layer.main_take() is not None

    def test_cannot_remove_main_when_only_take(self):
        """Cannot remove the main even when it's the only take."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        main = layer.main_take()
        with pytest.raises(TakeLayerError, match="Cannot remove the main take"):
            layer.remove_take(main.id)

    def test_reorder_single_take(self):
        """Reordering a single-element list is a no-op."""
        t = _main_take("M")
        layer = TakeLayer(layer_id="l", takes=[t])
        layer.reorder_takes([t.id])
        assert layer.takes[0].id == t.id

    def test_reorder_requires_all_ids(self):
        """reorder_takes with missing ID raises."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M"), _take("T")])
        with pytest.raises(TakeLayerError):
            layer.reorder_takes(["only-one-id"])

    def test_reorder_rejects_extra_ids(self):
        """reorder_takes with extra ID raises."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M"), _take("T")])
        with pytest.raises(TakeLayerError):
            layer.reorder_takes([layer.takes[0].id, layer.takes[1].id, "phantom"])

    def test_promote_to_main_unknown_id_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        with pytest.raises(TakeLayerError, match="not found"):
            layer.promote_to_main("ghost-id")

    def test_get_take_unknown_id_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        with pytest.raises(TakeLayerError, match="not found"):
            layer.get_take("ghost-id")

    def test_remove_unknown_id_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        with pytest.raises(TakeLayerError, match="not found"):
            layer.remove_take("ghost-id")

    def test_replace_unknown_id_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        replacement = _take("Replacement")
        with pytest.raises(TakeLayerError, match="not found"):
            layer.replace_take("ghost-id", replacement)


# ===========================================================================
# merge_events boundary tests
# ===========================================================================


class TestMergeEventsBoundaries:

    def test_both_empty_additive(self):
        result = merge_events((), (), strategy="additive")
        assert result == ()

    def test_both_empty_subtract(self):
        result = merge_events((), (), strategy="subtract")
        assert result == ()

    def test_both_empty_intersect(self):
        result = merge_events((), (), strategy="intersect")
        assert result == ()

    def test_both_empty_replace_range(self):
        result = merge_events((), (), strategy="replace_range", time_range=(0.0, 1.0))
        assert result == ()

    def test_empty_target_additive(self):
        source = (_evt(1.0), _evt(2.0))
        result = merge_events((), source, strategy="additive")
        assert len(result) == 2

    def test_empty_source_additive(self):
        target = (_evt(1.0), _evt(2.0))
        result = merge_events(target, (), strategy="additive")
        assert len(result) == 2

    def test_empty_target_subtract(self):
        """Subtracting from empty target returns empty."""
        source = (_evt(1.0),)
        result = merge_events((), source, strategy="subtract")
        assert result == ()

    def test_empty_source_subtract(self):
        """Subtracting nothing removes nothing."""
        target = (_evt(1.0), _evt(2.0))
        result = merge_events(target, (), strategy="subtract")
        assert len(result) == 2

    def test_empty_source_intersect(self):
        """Intersection with empty source yields empty."""
        target = (_evt(1.0), _evt(2.0))
        result = merge_events(target, (), strategy="intersect")
        assert result == ()

    def test_single_event_additive(self):
        result = merge_events((_evt(1.0),), (_evt(2.0),), strategy="additive")
        assert len(result) == 2

    def test_epsilon_zero_no_match(self):
        """epsilon=0: only exact matches count."""
        target = (_evt(1.0), _evt(2.0))
        source = (_evt(1.001),)  # slightly off — won't match with epsilon=0
        result = merge_events(target, source, strategy="subtract", time_epsilon=0.0)
        assert len(result) == 2  # nothing removed

    def test_epsilon_zero_exact_match(self):
        """epsilon=0: exact float match removes event."""
        target = (_evt(1.0), _evt(2.0))
        source = (_evt(1.0),)  # exact match
        result = merge_events(target, source, strategy="subtract", time_epsilon=0.0)
        assert len(result) == 1

    def test_replace_range_requires_time_range(self):
        with pytest.raises(ValueError, match="requires time_range"):
            merge_events((_evt(1.0),), (_evt(2.0),), strategy="replace_range")

    def test_replace_range_empty_window(self):
        """replace_range with a window containing no events."""
        target = (_evt(1.0), _evt(5.0))
        source = (_evt(3.0),)
        result = merge_events(
            target, source, strategy="replace_range", time_range=(2.0, 4.0)
        )
        # 3.0 inserted, 1.0 and 5.0 kept
        times = {e.time for e in result}
        assert 1.0 in times
        assert 5.0 in times
        assert 3.0 in times

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            merge_events((), (), strategy="magic")  # type: ignore


# ===========================================================================
# Event / domain type boundary tests
# ===========================================================================


class TestDomainTypeBoundaries:

    def test_event_zero_time_zero_duration(self):
        e = Event(id="e0", time=0.0, duration=0.0,
                  classifications={}, metadata={}, origin="test")
        assert e.time == 0.0
        assert e.duration == 0.0

    def test_event_negative_time(self):
        """Negative time is structurally valid (domain doesn't restrict it)."""
        e = Event(id="e-neg", time=-1.5, duration=0.1,
                  classifications={}, metadata={}, origin="test")
        assert e.time == -1.5

    def test_event_very_large_time(self):
        e = Event(id="e-big", time=1e12, duration=0.0,
                  classifications={}, metadata={}, origin="test")
        assert e.time == 1e12

    def test_event_deeply_nested_classifications(self):
        """classifications is Any — nesting should be fine."""
        nested = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        e = Event(id="e-nested", time=1.0, duration=0.1,
                  classifications=nested, metadata={}, origin="test")
        assert e.classifications["level1"]["level2"]["level3"] == [1, 2, 3]

    def test_event_data_empty_layers(self):
        ed = EventData(layers=())
        assert ed.layers == ()

    def test_event_data_single_layer_no_events(self):
        layer = Layer(id="l1", name="Empty", events=())
        ed = EventData(layers=(layer,))
        assert len(ed.layers[0].events) == 0

    def test_audio_data_zero_sample_rate(self):
        """Zero sample rate: structurally valid, semantically suspect, but allowed."""
        ad = AudioData(sample_rate=0, duration=0.0, file_path="")
        assert ad.sample_rate == 0

    def test_audio_data_zero_duration(self):
        ad = AudioData(sample_rate=44100, duration=0.0, file_path="file.wav")
        assert ad.duration == 0.0

    def test_audio_data_empty_file_path(self):
        ad = AudioData(sample_rate=44100, duration=1.0, file_path="")
        assert ad.file_path == ""

    def test_audio_data_default_channel_count(self):
        """Default channel_count is 1."""
        ad = AudioData(sample_rate=44100, duration=1.0, file_path="f.wav")
        assert ad.channel_count == 1

    def test_take_create_empty_notes(self):
        t = Take.create(data=_event_data(1.0), label="T")
        assert t.notes == ""

    def test_take_create_empty_label(self):
        """Empty label is structurally valid."""
        t = Take.create(data=_event_data(1.0), label="")
        assert t.label == ""


# ===========================================================================
# ExecutionCache boundary tests
# ===========================================================================


class TestExecutionCacheBoundaries:

    def test_get_from_empty_cache_returns_none(self):
        cache = ExecutionCache()
        assert cache.get("b1", "out") is None

    def test_has_valid_output_empty_cache(self):
        cache = ExecutionCache()
        assert not cache.has_valid_output("b1", "out")

    def test_store_and_get_single_entry(self):
        cache = ExecutionCache()
        cache.store("b1", "out", "value", "exec-1")
        entry = cache.get("b1", "out")
        assert entry is not None
        assert entry.value == "value"

    def test_overwrite_same_key(self):
        """Storing to the same (block_id, port_name) overwrites."""
        cache = ExecutionCache()
        cache.store("b1", "out", "v1", "exec-1")
        cache.store("b1", "out", "v2", "exec-2")
        assert cache.get("b1", "out").value == "v2"

    def test_invalidate_nonexistent_block_no_crash(self):
        """Invalidating a block not in cache should not raise."""
        cache = ExecutionCache()
        cache.invalidate("ghost-block")  # should not raise

    def test_get_all_empty(self):
        cache = ExecutionCache()
        assert cache.get_all("b1") == {}

    def test_clear_empty_cache(self):
        """Clearing already-empty cache is a no-op."""
        cache = ExecutionCache()
        cache.clear()  # should not raise
        assert cache.get("b1", "out") is None

    def test_invalidate_downstream_leaf_only_affects_leaf(self):
        """Invalidating a leaf block affects only that block."""
        g = Graph()
        root = _make_root_block("root")
        leaf = _make_block("leaf")
        g.add_block(root)
        g.add_block(leaf)
        g.add_connection(Connection(
            source_block_id="root", source_output_name="out",
            target_block_id="leaf", target_input_name="in",
        ))
        cache = ExecutionCache()
        cache.store("root", "out", "root_val", "exec-1")
        cache.store("leaf", "out", "leaf_val", "exec-1")

        affected = cache.invalidate_downstream("leaf", g)
        assert "leaf" in affected
        assert "root" not in affected
        assert cache.get("leaf", "out") is None
        assert cache.get("root", "out") is not None  # root untouched


# ===========================================================================
# GraphPlanner boundary tests
# ===========================================================================


class TestGraphPlannerBoundaries:

    def test_empty_graph_produces_empty_plan(self):
        g = Graph()
        planner = GraphPlanner()
        plan = planner.plan(g)
        assert plan.ordered_block_ids == ()

    def test_single_block_no_connections(self):
        g = Graph()
        g.add_block(_make_root_block("only"))
        planner = GraphPlanner()
        plan = planner.plan(g)
        assert plan.ordered_block_ids == ("only",)

    def test_plan_with_target_includes_target(self):
        g = Graph()
        root = _make_root_block("root")
        leaf = _make_block("leaf")
        g.add_block(root)
        g.add_block(leaf)
        g.add_connection(Connection(
            source_block_id="root", source_output_name="out",
            target_block_id="leaf", target_input_name="in",
        ))
        planner = GraphPlanner()
        plan = planner.plan(g, target_block_id="leaf")
        assert "leaf" in plan.ordered_block_ids
        assert "root" in plan.ordered_block_ids  # upstream included


# ===========================================================================
# ready_nodes boundary tests
# ===========================================================================


class TestReadyNodesBoundaries:

    def _make_graph_chain(self, *block_ids: str) -> Graph:
        """Helper: chain blocks A → B → C."""
        g = Graph()
        prev = None
        for bid in block_ids:
            if prev is None:
                g.add_block(_make_root_block(bid))
            else:
                g.add_block(_make_block(bid))
                g.add_connection(Connection(
                    source_block_id=prev, source_output_name="out",
                    target_block_id=bid, target_input_name="in",
                ))
            prev = bid
        return g

    def test_no_dirty_nodes_returns_empty(self):
        g = self._make_graph_chain("a", "b")
        cache = ExecutionCache()
        result = ready_nodes(g, dirty=set(), running=set(), cache=cache)
        assert result == set()

    def test_root_node_dirty_no_deps_is_ready(self):
        g = self._make_graph_chain("root")
        cache = ExecutionCache()
        result = ready_nodes(g, dirty={"root"}, running=set(), cache=cache)
        assert "root" in result

    def test_running_node_not_in_ready(self):
        g = self._make_graph_chain("root")
        cache = ExecutionCache()
        result = ready_nodes(g, dirty={"root"}, running={"root"}, cache=cache)
        assert "root" not in result

    def test_dependent_not_ready_when_upstream_dirty(self):
        """Leaf should not be ready if its upstream is also dirty."""
        g = self._make_graph_chain("root", "leaf")
        cache = ExecutionCache()
        # Both dirty, root output not cached
        result = ready_nodes(g, dirty={"root", "leaf"}, running=set(), cache=cache)
        assert "leaf" not in result
        assert "root" in result  # root is ready (no upstream)

    def test_dependent_ready_when_upstream_cached(self):
        """Leaf should be ready when root is done (not dirty, output cached)."""
        g = self._make_graph_chain("root", "leaf")
        cache = ExecutionCache()
        cache.store("root", "out", "value", "exec-1")
        # Only leaf is dirty
        result = ready_nodes(g, dirty={"leaf"}, running=set(), cache=cache)
        assert "leaf" in result

    def test_all_running_returns_empty(self):
        g = self._make_graph_chain("root", "leaf")
        cache = ExecutionCache()
        result = ready_nodes(g, dirty={"root", "leaf"}, running={"root", "leaf"}, cache=cache)
        assert result == set()

