"""
Anomaly / Error Path Tests for EchoZero 2.

SQLite principle: test failure paths as rigorously as happy paths.
The system must degrade gracefully — return Err, raise the documented exception,
or no-op — never silently corrupt state or crash with an undocumented exception.

Covers:
- ExecutionEngine: missing block, no executor, executor exception, executor Err, cancel
- Coordinator: request_run failure leaves is_executing=False, cancel idempotent
- ExecutionCache: invalidate unknown, invalidate downstream on unknown
- Graph: cycle detection, duplicate block, bad port type, self-connection, missing block
- Serialization: corrupted/missing fields raise predictably
- TakeLayer: illegal state transitions (restore corrupt snapshot)
- Pipeline: unregistered command returns Err (not raise)
- merge_take_into: AudioData raises, missing source/target IDs
"""

import json
import threading
import tempfile
import os

import pytest

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
from echozero.editor.cache import ExecutionCache
from echozero.editor.coordinator import Coordinator, ready_nodes
from echozero.errors import ExecutionError, OperationCancelledError, ValidationError
from echozero.event_bus import EventBus
from echozero.execution import ExecutionEngine, ExecutionPlan, GraphPlanner
from echozero.editor.pipeline import Pipeline
from echozero.progress import RuntimeBus
from echozero.result import Err, Ok, err, is_err, is_ok, ok
from echozero.serialization import (
    deserialize_graph,
    deserialize_take,
    deserialize_take_layer,
    load_project,
)
from echozero.takes import (
    Take,
    TakeLayer,
    TakeLayerError,
    TakeLayerSnapshot,
    TakeSource,
    merge_take_into,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(time: float) -> Event:
    return Event(id=f"e-{time}", time=time, duration=0.1,
                 classifications={}, metadata={}, origin="test")


def _event_data(*times: float) -> EventData:
    events = tuple(_evt(t) for t in times)
    layer = Layer(id="l1", name="L", events=events)
    return EventData(layers=(layer,))


def _main_take(label: str = "Main", *times) -> Take:
    if not times:
        times = (1.0,)
    return Take.create(data=_event_data(*times), label=label, is_main=True)


def _take(label: str = "T", *times) -> Take:
    if not times:
        times = (1.0,)
    return Take.create(data=_event_data(*times), label=label, is_main=False)


def _make_root_block(bid: str) -> Block:
    return Block(
        id=bid, name=bid, block_type="source",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
    )


def _make_block(bid: str, btype: str = "proc") -> Block:
    return Block(
        id=bid, name=bid, block_type=btype,
        category=BlockCategory.PROCESSOR,
        input_ports=(Port(name="in", port_type=PortType.EVENT, direction=Direction.INPUT),),
        output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
    )


class OkExecutor:
    """Always returns Ok with a value."""
    def __init__(self, value="output"):
        self._value = value

    def execute(self, block_id, context):
        return ok(self._value)


class ErrExecutor:
    """Always returns an Err."""
    def __init__(self, msg="executor failed"):
        self._msg = msg

    def execute(self, block_id, context):
        return err(ExecutionError(self._msg))


class BoomExecutor:
    """Raises an exception (simulates unexpected crash)."""
    def execute(self, block_id, context):
        raise RuntimeError("unexpected executor crash")


class SlowCancelExecutor:
    """Checks cancel_event mid-execution."""
    def __init__(self, cancel_event: threading.Event):
        self._cancel = cancel_event

    def execute(self, block_id, context):
        self._cancel.set()  # trigger cancel from inside executor
        return ok("done")


def _make_engine_with_graph(graph: Graph, bus: RuntimeBus = None) -> ExecutionEngine:
    return ExecutionEngine(graph, bus or RuntimeBus())


# ===========================================================================
# ExecutionEngine anomaly tests
# ===========================================================================


class TestExecutionEngineAnomalies:

    def test_plan_references_missing_block_returns_err(self):
        """Block in plan but not in graph → Err, not KeyError."""
        g = Graph()
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        plan = ExecutionPlan(execution_id="x", ordered_block_ids=("ghost",))
        result = engine.run(plan)
        assert is_err(result)

    def test_no_executor_registered_returns_err(self):
        """Block exists but has no registered executor → Err."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        engine = _make_engine_with_graph(g)
        # No executor registered for "source"
        planner = GraphPlanner()
        plan = planner.plan(g)
        result = engine.run(plan)
        assert is_err(result)

    def test_executor_returns_err_propagates(self):
        """Executor returning Err → engine returns Err."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        engine = _make_engine_with_graph(g)
        engine.register_executor("source", ErrExecutor("deliberate failure"))
        planner = GraphPlanner()
        plan = planner.plan(g)
        result = engine.run(plan)
        assert is_err(result)
        assert "deliberate failure" in str(result.error)

    def test_executor_exception_returns_err_not_raise(self):
        """Executor raising an exception → engine returns Err (not raise)."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        engine = _make_engine_with_graph(g)
        engine.register_executor("source", BoomExecutor())
        planner = GraphPlanner()
        plan = planner.plan(g)
        result = engine.run(plan)
        assert is_err(result)

    def test_cancel_before_run_returns_cancelled_err(self):
        """Cancel event set before run → immediately returns OperationCancelledError."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        engine = _make_engine_with_graph(g)
        engine.register_executor("source", OkExecutor())
        planner = GraphPlanner()
        plan = planner.plan(g)
        cancel = threading.Event()
        cancel.set()  # already cancelled
        result = engine.run(plan, cancel_event=cancel)
        assert is_err(result)
        assert isinstance(result.error, OperationCancelledError)

    def test_cancel_between_blocks_stops_execution(self):
        """First block runs, cancel fires, second block never runs."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        g.add_block(_make_block("b2"))
        g.add_connection(Connection(
            source_block_id="b1", source_output_name="out",
            target_block_id="b2", target_input_name="in",
        ))

        cancel = threading.Event()
        executed = []

        class RecordingExecutor:
            def __init__(self, bid):
                self._bid = bid
            def execute(self, block_id, context):
                executed.append(self._bid)
                if self._bid == "b1":
                    cancel.set()  # cancel after first block
                return ok("val")

        engine = _make_engine_with_graph(g)
        engine.register_executor("source", RecordingExecutor("b1"))
        engine.register_executor("proc", RecordingExecutor("b2"))

        planner = GraphPlanner()
        plan = planner.plan(g)
        result = engine.run(plan, cancel_event=cancel)

        # b1 ran, b2 did not (cancel fired between them)
        assert is_err(result)
        assert "b1" in executed
        assert "b2" not in executed

    def test_first_block_err_stops_chain(self):
        """When first block fails, downstream blocks never execute."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        g.add_block(_make_block("b2"))
        g.add_connection(Connection(
            source_block_id="b1", source_output_name="out",
            target_block_id="b2", target_input_name="in",
        ))

        executed = []

        class TrackingExecutor:
            def __init__(self, bid, should_fail=False):
                self._bid = bid
                self._fail = should_fail
            def execute(self, block_id, context):
                executed.append(self._bid)
                if self._fail:
                    return err(ExecutionError("first block failed"))
                return ok("val")

        engine = _make_engine_with_graph(g)
        engine.register_executor("source", TrackingExecutor("b1", should_fail=True))
        engine.register_executor("proc", TrackingExecutor("b2"))

        planner = GraphPlanner()
        plan = planner.plan(g)
        result = engine.run(plan)

        assert is_err(result)
        assert "b1" in executed
        assert "b2" not in executed

    def test_empty_plan_succeeds_with_empty_outputs(self):
        """Empty plan (0 blocks) returns Ok with empty dict."""
        g = Graph()
        engine = _make_engine_with_graph(g)
        plan = ExecutionPlan(execution_id="empty", ordered_block_ids=())
        result = engine.run(plan)
        assert is_ok(result)
        assert result.value == {}


# ===========================================================================
# Coordinator anomaly tests
# ===========================================================================


class TestCoordinatorAnomalies:

    def _make_coordinator(self, graph: Graph) -> Coordinator:
        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        cache = ExecutionCache()
        pipeline = Pipeline(EventBus())
        return Coordinator(
            graph=graph,
            pipeline=pipeline,
            engine=engine,
            cache=cache,
            runtime_bus=bus,
        )

    def test_is_executing_false_after_failed_run(self):
        """Even if execution fails, is_executing returns to False."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        coord = self._make_coordinator(g)
        # No executor registered → execution will fail
        result = coord.request_run()
        assert is_err(result)
        assert not coord.is_executing

    def test_is_executing_false_after_successful_run(self):
        """After successful run, is_executing is False."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        coord = self._make_coordinator(g)
        coord._engine.register_executor("source", OkExecutor())
        result = coord.request_run()
        assert is_ok(result)
        assert not coord.is_executing

    def test_cancel_when_not_running_is_idempotent(self):
        """Calling cancel() when nothing is running should not raise."""
        g = Graph()
        coord = self._make_coordinator(g)
        coord.cancel()  # should not raise
        coord.cancel()  # double cancel — also fine

    def test_propagate_stale_marks_downstream(self):
        """propagate_stale on a root block marks all downstream as STALE."""
        g = Graph()
        g.add_block(_make_root_block("root"))
        g.add_block(_make_block("leaf"))
        g.add_connection(Connection(
            source_block_id="root", source_output_name="out",
            target_block_id="leaf", target_input_name="in",
        ))
        coord = self._make_coordinator(g)
        # Manually set leaf to FRESH
        g.set_block_state("leaf", BlockState.FRESH)
        coord.propagate_stale("root")
        assert g.blocks["leaf"].state == BlockState.STALE

    def test_request_run_returns_execution_id_string(self):
        """On success, request_run returns Ok with a non-empty string."""
        g = Graph()
        g.add_block(_make_root_block("b1"))
        coord = self._make_coordinator(g)
        coord._engine.register_executor("source", OkExecutor())
        result = coord.request_run()
        assert is_ok(result)
        assert isinstance(result.value, str)
        assert len(result.value) > 0


# ===========================================================================
# ExecutionCache anomaly tests
# ===========================================================================


class TestExecutionCacheAnomalies:

    def test_invalidate_nonexistent_block_no_crash(self):
        cache = ExecutionCache()
        cache.invalidate("does-not-exist")  # must not raise

    def test_invalidate_downstream_nonexistent_block_raises_validation(self):
        """Graph.downstream_of raises ValidationError for unknown block."""
        cache = ExecutionCache()
        g = Graph()
        with pytest.raises(ValidationError):
            cache.invalidate_downstream("ghost", g)

    def test_get_all_for_unknown_block_returns_empty(self):
        cache = ExecutionCache()
        result = cache.get_all("unknown")
        assert result == {}

    def test_double_clear_is_safe(self):
        cache = ExecutionCache()
        cache.store("b1", "out", "val", "exec-1")
        cache.clear()
        cache.clear()  # second clear should not crash
        assert cache.get("b1", "out") is None


# ===========================================================================
# Graph anomaly tests
# ===========================================================================


class TestGraphAnomalies:

    def test_add_duplicate_block_raises(self):
        g = Graph()
        b = _make_root_block("b1")
        g.add_block(b)
        with pytest.raises(ValidationError, match="Duplicate block ID"):
            g.add_block(b)

    def test_self_connection_raises(self):
        g = Graph()
        # Need a block that has both in and out ports
        b = Block(
            id="b1", name="B1", block_type="proc",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port(name="in", port_type=PortType.EVENT, direction=Direction.INPUT),),
            output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
        )
        g.add_block(b)
        with pytest.raises(ValidationError, match="Self-connections"):
            g.add_connection(Connection(
                source_block_id="b1", source_output_name="out",
                target_block_id="b1", target_input_name="in",
            ))

    def test_cycle_detection_raises(self):
        """A → B → C → A should be rejected."""
        g = Graph()
        # Make blocks with both in and out ports
        def _bidir_block(bid):
            return Block(
                id=bid, name=bid, block_type="proc",
                category=BlockCategory.PROCESSOR,
                input_ports=(Port(name="in", port_type=PortType.EVENT, direction=Direction.INPUT),),
                output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
            )
        g.add_block(_bidir_block("a"))
        g.add_block(_bidir_block("b"))
        g.add_connection(Connection(
            source_block_id="a", source_output_name="out",
            target_block_id="b", target_input_name="in",
        ))
        with pytest.raises(ValidationError, match="cycle"):
            g.add_connection(Connection(
                source_block_id="b", source_output_name="out",
                target_block_id="a", target_input_name="in",
            ))

    def test_connection_missing_source_block_raises(self):
        g = Graph()
        g.add_block(_make_block("b2"))
        with pytest.raises(ValidationError, match="Source block not found"):
            g.add_connection(Connection(
                source_block_id="ghost", source_output_name="out",
                target_block_id="b2", target_input_name="in",
            ))

    def test_connection_missing_target_block_raises(self):
        g = Graph()
        g.add_block(_make_root_block("b1"))
        with pytest.raises(ValidationError, match="Target block not found"):
            g.add_connection(Connection(
                source_block_id="b1", source_output_name="out",
                target_block_id="ghost", target_input_name="in",
            ))

    def test_port_type_mismatch_raises(self):
        """EVENT output → AUDIO input should be rejected."""
        event_out_block = Block(
            id="event_src", name="EventSrc", block_type="src",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port(name="out", port_type=PortType.EVENT, direction=Direction.OUTPUT),),
        )
        audio_in_block = Block(
            id="audio_sink", name="AudioSink", block_type="sink",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port(name="in", port_type=PortType.AUDIO, direction=Direction.INPUT),),
            output_ports=(),
        )
        g = Graph()
        g.add_block(event_out_block)
        g.add_block(audio_in_block)
        with pytest.raises(ValidationError, match="Port type mismatch"):
            g.add_connection(Connection(
                source_block_id="event_src", source_output_name="out",
                target_block_id="audio_sink", target_input_name="in",
            ))

    def test_set_state_unknown_block_raises(self):
        g = Graph()
        with pytest.raises(ValidationError):
            g.set_block_state("ghost", BlockState.FRESH)

    def test_remove_unknown_block_raises(self):
        g = Graph()
        with pytest.raises(ValidationError):
            g.remove_block("ghost")

    def test_downstream_of_unknown_block_raises(self):
        g = Graph()
        with pytest.raises(ValidationError):
            g.downstream_of("ghost")

    def test_remove_connection_not_found_raises(self):
        g = Graph()
        g.add_block(_make_root_block("b1"))
        g.add_block(_make_block("b2"))
        conn = Connection(
            source_block_id="b1", source_output_name="out",
            target_block_id="b2", target_input_name="in",
        )
        with pytest.raises(ValidationError, match="Connection not found"):
            g.remove_connection(conn)


# ===========================================================================
# Serialization anomaly tests
# ===========================================================================


class TestSerializationAnomalies:

    def test_deserialize_take_unknown_data_type_raises(self):
        """Unknown 'type' in take data → ValueError."""
        data = {
            "id": "t1",
            "label": "Test",
            "origin": "pipeline",
            "created_at": "2024-01-01T00:00:00+00:00",
            "is_main": True,
            "notes": "",
            "source": None,
            "data": {"type": "UnknownDataType", "stuff": 123},
        }
        with pytest.raises(ValueError, match="Unknown take data type"):
            deserialize_take(data)

    def test_deserialize_take_missing_required_field_raises(self):
        """Missing 'id' → KeyError."""
        data = {
            "label": "Test",
            "origin": "pipeline",
            "created_at": "2024-01-01T00:00:00+00:00",
            "is_main": True,
            "notes": "",
            "source": None,
            "data": {"type": "EventData", "layers": []},
        }
        with pytest.raises(KeyError):
            deserialize_take(data)

    def test_deserialize_graph_missing_blocks_key_raises(self):
        """Missing 'blocks' key → KeyError."""
        with pytest.raises(KeyError):
            deserialize_graph({"connections": []})

    def test_deserialize_graph_unknown_enum_raises(self):
        """Unknown BlockCategory → KeyError."""
        data = {
            "blocks": [{
                "id": "b1",
                "name": "B",
                "block_type": "x",
                "category": "INVALID_CATEGORY",
                "state": "FRESH",
                "input_ports": [],
                "output_ports": [],
                "control_ports": [],
                "settings": {},
            }],
            "connections": [],
        }
        with pytest.raises(KeyError):
            deserialize_graph(data)

    def test_load_project_invalid_json_raises(self, tmp_path):
        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("{ this is not valid JSON !!!")
        with pytest.raises(Exception):  # json.JSONDecodeError subclass
            load_project(path)

    def test_load_project_missing_file_raises(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            load_project(path)

    def test_load_project_missing_graph_key_raises(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump({"version": "2.1.0", "take_layers": []}, f)
        with pytest.raises(KeyError):
            load_project(path)

    def test_deserialize_take_layer_no_main_raises(self):
        """Deserializing a TakeLayer with no main take → TakeLayerError."""
        data = {
            "layer_id": "l1",
            "takes": [{
                "id": "t1",
                "label": "Take",
                "origin": "pipeline",
                "created_at": "2024-01-01T00:00:00+00:00",
                "is_main": False,  # ← no main!
                "notes": "",
                "source": None,
                "data": {"type": "EventData", "layers": []},
            }],
        }
        with pytest.raises(TakeLayerError, match="0 main takes"):
            deserialize_take_layer(data)


# ===========================================================================
# TakeLayer illegal state transitions
# ===========================================================================


class TestTakeLayerAnomalies:

    def test_restore_corrupt_snapshot_zero_mains_raises(self):
        """Restoring a snapshot with no main take violates invariant."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        # Craft a corrupt snapshot (all takes have is_main=False)
        non_main_take = _take("T")
        corrupt_snap = TakeLayerSnapshot(
            layer_id="l",
            takes=(non_main_take,),  # no main
        )
        with pytest.raises(TakeLayerError, match="0 main takes"):
            layer.restore(corrupt_snap)

    def test_restore_corrupt_snapshot_two_mains_raises(self):
        """Restoring a snapshot with two mains violates invariant."""
        from dataclasses import replace
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        m1 = _main_take("M1")
        m2 = _main_take("M2")
        corrupt_snap = TakeLayerSnapshot(layer_id="l", takes=(m1, m2))
        with pytest.raises(TakeLayerError, match="2 main takes"):
            layer.restore(corrupt_snap)

    def test_add_take_with_main_flag_raises(self):
        """Adding a take with is_main=True via add_take() is rejected."""
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        bad = _main_take("Also Main")
        with pytest.raises(TakeLayerError, match="must not be main"):
            layer.add_take(bad)

    def test_promote_nonexistent_id_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M")])
        with pytest.raises(TakeLayerError, match="not found"):
            layer.promote_to_main("does-not-exist")


# ===========================================================================
# merge_take_into anomaly tests
# ===========================================================================


class TestMergeTakeIntoAnomalies:

    def test_merge_audio_take_raises(self):
        audio = AudioData(sample_rate=44100, duration=3.0, file_path="f.wav")
        main = Take.create(data=audio, label="Audio Main", origin="sync", is_main=True)
        src = Take.create(data=audio, label="Audio Src", origin="pipeline")
        layer = TakeLayer(layer_id="l", takes=[main])
        layer.add_take(src)
        with pytest.raises(TakeLayerError, match="only supported for EventData"):
            merge_take_into(layer, src.id, main.id)

    def test_merge_nonexistent_source_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M", 1.0)])
        with pytest.raises(TakeLayerError, match="not found"):
            merge_take_into(layer, "ghost-source", layer.main_take().id)

    def test_merge_nonexistent_target_raises(self):
        layer = TakeLayer(layer_id="l", takes=[_main_take("M", 1.0)])
        src = _take("Src", 2.0)
        layer.add_take(src)
        with pytest.raises(TakeLayerError, match="not found"):
            merge_take_into(layer, src.id, "ghost-target")


# ===========================================================================
# Pipeline anomaly tests
# ===========================================================================


class TestPipelineAnomalies:

    def test_dispatch_unregistered_command_returns_err(self):
        """Dispatching a command with no handler returns Err, not raise."""
        from echozero.editor.commands import Command
        from dataclasses import dataclass

        # Command base is frozen=True; subclass must also be frozen=True
        @dataclass(frozen=True)
        class UnknownCommand(Command):
            value: int = 0

            @property
            def is_undoable(self) -> bool:
                return False

        pipe = Pipeline(EventBus())
        result = pipe.dispatch(UnknownCommand(value=42))
        assert is_err(result)

    def test_dispatch_handler_exception_returns_err(self):
        """If handler raises, pipeline returns Err (not propagate)."""
        from echozero.editor.commands import Command
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class BoomCommand(Command):
            @property
            def is_undoable(self) -> bool:
                return False

        def boom_handler(cmd, ctx):
            raise ValueError("handler blew up")

        pipe = Pipeline(EventBus())
        pipe.register(BoomCommand, boom_handler)
        result = pipe.dispatch(BoomCommand())
        assert is_err(result)

