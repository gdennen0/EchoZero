"""Tests for DatasetViewerProcessor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Port
from echozero.execution import ExecutionEngine, GraphPlanner
from echozero.processors.dataset_viewer import DatasetViewerProcessor, AUDIO_EXTENSIONS
from echozero.progress import RuntimeBus
from echozero.result import is_err, is_ok, unwrap


def _fake_scan(dataset_dir, audio_extensions):
    return {
        "dataset_dir": dataset_dir,
        "total_files": 15,
        "total_classes": 3,
        "classes": {
            "kick": {"count": 8, "total_bytes": 40000},
            "snare": {"count": 5, "total_bytes": 25000},
            "hihat": {"count": 2, "total_bytes": 10000},
        },
    }


def _empty_scan(dataset_dir, audio_extensions):
    return {
        "dataset_dir": dataset_dir,
        "total_files": 0,
        "total_classes": 0,
        "classes": {},
    }


def _build_graph(dataset_dir="/tmp/dataset") -> Graph:
    g = Graph()
    g.add_block(Block(
        id="viewer", name="Viewer", block_type="DatasetViewer",
        category=BlockCategory.WORKSPACE,
        input_ports=(), output_ports=(),
        settings=BlockSettings({"dataset_dir": dataset_dir}),
    ))
    return g


def _run(graph, scan_fn=_fake_scan):
    bus = RuntimeBus()
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("DatasetViewer", DatasetViewerProcessor(scan_fn))
    plan = GraphPlanner().plan(graph)
    return engine.run(plan)


class TestDatasetViewerProcessor:
    def test_scans_successfully(self):
        result = _run(_build_graph())
        assert is_ok(result)

    def test_returns_total_files(self):
        result = _run(_build_graph())
        stats = unwrap(result)["viewer"]["out"]
        assert stats["total_files"] == 15

    def test_returns_total_classes(self):
        result = _run(_build_graph())
        stats = unwrap(result)["viewer"]["out"]
        assert stats["total_classes"] == 3

    def test_returns_class_breakdown(self):
        result = _run(_build_graph())
        stats = unwrap(result)["viewer"]["out"]
        assert "kick" in stats["classes"]
        assert stats["classes"]["kick"]["count"] == 8

    def test_empty_dataset(self):
        result = _run(_build_graph(), _empty_scan)
        assert is_ok(result)
        stats = unwrap(result)["viewer"]["out"]
        assert stats["total_files"] == 0

    def test_missing_dataset_dir_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="viewer", name="V", block_type="DatasetViewer",
            category=BlockCategory.WORKSPACE,
            input_ports=(), output_ports=(),
            settings=BlockSettings({}),
        ))
        result = _run(g)
        assert is_err(result)

    def test_scan_failure_returns_error(self):
        def failing(*args, **kwargs):
            raise RuntimeError("permission denied")
        result = _run(_build_graph(), failing)
        assert is_err(result)

    def test_audio_extensions_constant(self):
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".flac" in AUDIO_EXTENSIONS

    def test_dataset_dir_passed_to_scan(self):
        called_with = []

        def spy_scan(dataset_dir, audio_extensions):
            called_with.append(dataset_dir)
            return _fake_scan(dataset_dir, audio_extensions)

        _run(_build_graph(dataset_dir="/my/data"), spy_scan)
        assert called_with == ["/my/data"]


