"""
Tests for EditorGetLayersCommand.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.commands.editor_commands import EditorGetLayersCommand
from src.shared.domain.entities import EventDataItem, EventLayer


class _DataItemRepo:
    def __init__(self, items):
        self._items = items

    def list_by_block(self, block_id):
        return self._items


def _make_facade(items, overrides):
    facade = MagicMock()
    facade.data_item_repo = _DataItemRepo(items)
    facade.ui_state_repo = True

    def _get_ui_state(state_type, entity_id):
        return SimpleNamespace(success=True, data={"layers": overrides})

    facade.get_ui_state = MagicMock(side_effect=_get_ui_state)
    return facade


def test_get_layers_uses_event_items_and_overrides():
    event_layers = [EventLayer(name="kicks"), EventLayer(name="snares")]
    item = EventDataItem(
        id="item-1",
        block_id="block-1",
        name="Drums",
        layers=event_layers,
    )
    overrides = [
        {
            "name": "kicks",
            "group_id": "item-1",
            "height": 55,
            "color": "#ffffff",
        }
    ]
    facade = _make_facade([item], overrides)

    cmd = EditorGetLayersCommand(facade, "block-1")
    cmd.redo()

    assert len(cmd.layers) == 2
    layer_by_name = {layer["name"]: layer for layer in cmd.layers}
    assert layer_by_name["kicks"]["group_id"] == "item-1"
    assert layer_by_name["kicks"]["group_name"] == "Drums"
    assert layer_by_name["kicks"]["height"] == 55
    assert layer_by_name["kicks"]["color"] == "#ffffff"
    assert layer_by_name["snares"]["height"] == 40


def test_get_layers_raises_on_stale_override():
    event_layers = [EventLayer(name="kicks")]
    item = EventDataItem(
        id="item-1",
        block_id="block-1",
        name="Drums",
        layers=event_layers,
    )
    overrides = [{"name": "kicks", "group_id": "stale-group"}]
    facade = _make_facade([item], overrides)

    cmd = EditorGetLayersCommand(facade, "block-1")
    with pytest.raises(ValueError):
        cmd.redo()


def test_get_layers_uses_tc_group_id_when_present():
    event_layers = [EventLayer(name="tc_layer")]
    item = EventDataItem(
        id="item-2",
        block_id="block-1",
        name="MA3 Sync",
        metadata={"group_id": "tc_12", "group_name": "TC 12"},
        layers=event_layers,
    )
    facade = _make_facade([item], [])

    cmd = EditorGetLayersCommand(facade, "block-1")
    cmd.redo()

    assert cmd.layers[0]["group_id"] == "tc_12"
    assert cmd.layers[0]["group_name"] == "TC 12"
