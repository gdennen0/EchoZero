"""
Tests for LayerOrderService.
"""
from dataclasses import dataclass

from src.application.services.layer_order_service import LayerOrderService
from src.shared.domain.entities.layer_order import LayerOrder, LayerKey
from src.shared.domain.repositories.layer_order_repository import LayerOrderRepository


@dataclass
class MockLayer:
    group_id: str
    name: str


class InMemoryLayerOrderRepo(LayerOrderRepository):
    def __init__(self):
        self._store = {}

    def get_order(self, block_id: str):
        return self._store.get(block_id)

    def set_order(self, layer_order: LayerOrder) -> None:
        self._store[layer_order.block_id] = layer_order

    def clear_order(self, block_id: str) -> None:
        self._store.pop(block_id, None)


def test_reconcile_when_no_saved_order():
    repo = InMemoryLayerOrderRepo()
    service = LayerOrderService(repo)
    block_id = "block-1"

    layers = [
        MockLayer(group_id="g1", name="a"),
        MockLayer(group_id="g1", name="b"),
    ]

    order = service.reconcile_and_save(block_id, layers)
    assert order == [LayerKey("g1", "a"), LayerKey("g1", "b")]
    assert repo.get_order(block_id) is not None


def test_append_missing_layers_within_group():
    repo = InMemoryLayerOrderRepo()
    service = LayerOrderService(repo)
    block_id = "block-2"

    saved = LayerOrder(
        block_id=block_id,
        order=[
            LayerKey("g1", "one"),
            LayerKey("g1", "two"),
            LayerKey("g2", "alpha"),
        ],
    )
    repo.set_order(saved)

    layers = [
        MockLayer(group_id="g1", name="one"),
        MockLayer(group_id="g1", name="two"),
        MockLayer(group_id="g1", name="new"),
        MockLayer(group_id="g2", name="alpha"),
    ]

    order = service.reconcile_and_save(block_id, layers)
    assert order == [
        LayerKey("g1", "one"),
        LayerKey("g1", "two"),
        LayerKey("g1", "new"),
        LayerKey("g2", "alpha"),
    ]


def test_append_new_group_at_end():
    repo = InMemoryLayerOrderRepo()
    service = LayerOrderService(repo)
    block_id = "block-3"

    repo.set_order(LayerOrder(block_id=block_id, order=[LayerKey("g1", "one")]))

    layers = [
        MockLayer(group_id="g1", name="one"),
        MockLayer(group_id="g2", name="beta"),
    ]

    order = service.reconcile_and_save(block_id, layers)
    assert order == [LayerKey("g1", "one"), LayerKey("g2", "beta")]


def test_filters_removed_layers():
    repo = InMemoryLayerOrderRepo()
    service = LayerOrderService(repo)
    block_id = "block-4"

    repo.set_order(LayerOrder(
        block_id=block_id,
        order=[LayerKey("g1", "one"), LayerKey("g1", "two")]
    ))

    layers = [MockLayer(group_id="g1", name="two")]

    order = service.reconcile_and_save(block_id, layers)
    assert order == [LayerKey("g1", "two")]
