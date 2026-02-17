"""
Layer Group Order Service

Stores group order and per-group layer order for timelines.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from src.shared.domain.entities.layer_order import LayerKey


class LayerGroupOrderService:
    """Persist and reconcile group + per-group layer ordering."""

    def __init__(self, ui_state_repo):
        self._ui_state_repo = ui_state_repo
        self._state_type = "editor_layer_group_order"

    def get_order(self, block_id: str) -> Tuple[List[str], Dict[str, List[str]]]:
        if not self._ui_state_repo:
            return [], {}
        data = self._ui_state_repo.get(self._state_type, block_id)
        if not data:
            return [], {}
        group_order = [g for g in data.get("group_order", []) if g]
        layers_by_group = {}
        raw_layers = data.get("layers_by_group", {}) or {}
        if isinstance(raw_layers, dict):
            for key, items in raw_layers.items():
                if not key:
                    continue
                layers_by_group[key] = [name for name in items or [] if name]
        return group_order, layers_by_group

    def save_order(self, block_id: str, group_order: List[str], layers_by_group: Dict[str, List[str]]) -> None:
        if not self._ui_state_repo:
            return
        payload = {
            "group_order": [g for g in group_order if g],
            "layers_by_group": {k: [n for n in v if n] for k, v in (layers_by_group or {}).items() if k},
        }
        self._ui_state_repo.set(self._state_type, block_id, payload)

    def build_from_layers(self, layers: Iterable[object]) -> Tuple[List[str], Dict[str, List[str]]]:
        group_order: List[str] = []
        layers_by_group: Dict[str, List[str]] = {}
        for layer in layers:
            group_key = getattr(layer, "group_id", None) or getattr(layer, "group_name", None)
            name = getattr(layer, "name", None)
            if not group_key or not name:
                continue
            if group_key not in layers_by_group:
                layers_by_group[group_key] = []
                group_order.append(group_key)
            if name not in layers_by_group[group_key]:
                layers_by_group[group_key].append(name)
        return group_order, layers_by_group

    def reconcile_and_save(self, block_id: str, layers: Iterable[object]) -> List[LayerKey]:
        current_group_order, current_layers_by_group = self.build_from_layers(layers)
        saved_group_order, saved_layers_by_group = self.get_order(block_id)

        if not current_group_order:
            return []

        # Reconcile group order
        final_group_order = [g for g in saved_group_order if g in current_group_order]
        for g in current_group_order:
            if g not in final_group_order:
                final_group_order.append(g)

        # Reconcile per-group layer order
        final_layers_by_group: Dict[str, List[str]] = {}
        for group_key in final_group_order:
            current_layers = current_layers_by_group.get(group_key, [])
            saved_layers = saved_layers_by_group.get(group_key, [])
            final_layers = [name for name in saved_layers if name in current_layers]
            for name in current_layers:
                if name not in final_layers:
                    final_layers.append(name)
            if final_layers:
                final_layers_by_group[group_key] = final_layers

        final_order = [
            LayerKey(group_name=group_key, name=layer_name)
            for group_key in final_group_order
            for layer_name in final_layers_by_group.get(group_key, [])
        ]

        self.save_order(block_id, final_group_order, final_layers_by_group)
        return final_order
