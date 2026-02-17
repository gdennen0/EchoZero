"""
Layer Order Service

Provides a simple, repo-backed API for layer ordering logic.
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Tuple, Optional

from src.shared.domain.entities.layer_order import LayerOrder, LayerKey
from src.shared.domain.repositories.layer_order_repository import LayerOrderRepository
from src.utils.message import Log


class LayerOrderService:
    """Orchestrates layer order persistence and reconciliation."""

    def __init__(self, layer_order_repo: LayerOrderRepository):
        self._repo = layer_order_repo

    def get_order(self, block_id: str) -> List[LayerKey]:
        stored = self._repo.get_order(block_id)
        return stored.order if stored else []

    def save_order(self, block_id: str, order: Iterable[LayerKey]) -> None:
        cleaned = self._dedupe([key for key in order if key.name])
        self._repo.set_order(LayerOrder(block_id=block_id, order=cleaned))

    def add_layer(
        self,
        block_id: str,
        group_name: Optional[str],
        layer_name: str,
        is_synced: bool = False
    ) -> List[LayerKey]:
        """Add a layer to order, preserving group and intra-group order."""
        if not layer_name:
            return self.get_order(block_id)

        saved_order = self.get_order(block_id)
        target = LayerKey(group_name=group_name, name=layer_name)

        normalized = []
        sync_prefix = self._sync_prefix(group_name, layer_name, is_synced)
        for key in self._dedupe(saved_order):
            if sync_prefix and key.name.startswith(sync_prefix) and key.group_name != group_name:
                normalized.append(LayerKey(group_name=group_name, name=key.name))
            else:
                normalized.append(key)

        # Remove existing target before re-inserting.
        current = [key for key in normalized if self._key_tuple(key) != self._key_tuple(target)]

        group_indices = [idx for idx, key in enumerate(current) if key.group_name == group_name]
        if group_indices:
            insert_at = group_indices[-1] + 1
        else:
            insert_at = len(current)

        final = current[:insert_at] + [target] + current[insert_at:]
        self.save_order(block_id, final)


        return final

    def remove_layer(self, block_id: str, group_name: Optional[str], layer_name: str) -> List[LayerKey]:
        """Remove a layer from the order and persist."""
        if not layer_name:
            return self.get_order(block_id)

        saved_order = self.get_order(block_id)
        target = LayerKey(group_name=group_name, name=layer_name)
        final = [key for key in saved_order if self._key_tuple(key) != self._key_tuple(target)]
        if final != saved_order:
            self.save_order(block_id, final)
        return final

    def append_to_group(self, block_id: str, group_name: Optional[str], layer_name: str) -> List[LayerKey]:
        """Backward compatible alias for add_layer()."""
        return self.add_layer(block_id, group_name, layer_name, is_synced=False)

    def remove_from_group(self, block_id: str, group_name: Optional[str], layer_name: str) -> List[LayerKey]:
        """Backward compatible alias for remove_layer()."""
        return self.remove_layer(block_id, group_name, layer_name)

    def reconcile_and_save(self, block_id: str, current_layers: Iterable[object]) -> List[LayerKey]:
        """
        Reconcile saved order with current layers and persist if changed.

        Current layers are the active TimelineLayer objects (or any objects
        with group_id and name attributes).
        """
        saved_order = self.get_order(block_id)
        current_keys = self._layers_to_keys(current_layers)
        group_name_to_id = {}
        for layer in current_layers:
            name = getattr(layer, "name", None)
            group_name = getattr(layer, "group_name", None)
            group_id = getattr(layer, "group_id", None)
            if name and group_name and group_id:
                group_name_to_id[(group_name, name)] = group_id
        if not current_keys:
            return saved_order
        final_order = self._reconcile_order(saved_order, current_keys, group_name_to_id)

        if final_order != saved_order:
            self.save_order(block_id, final_order)

        Log.debug(
            f"LayerOrderService: Reconciled order for block {block_id} "
            f"(saved={len(saved_order)}, current={len(current_keys)}, final={len(final_order)})"
        )

        return final_order

    def _reconcile_order(
        self,
        saved: List[LayerKey],
        current: List[LayerKey],
        group_name_to_id: Dict[Tuple[str, str], str]
    ) -> List[LayerKey]:
        """Return order filtered to current layers and append new ones by group."""
        if not current:
            return []

        current_set = {self._key_tuple(key) for key in current}
        current_by_name: Dict[str, List[Optional[str]]] = {}
        for key in current:
            current_by_name.setdefault(key.name, []).append(key.group_name)

        normalized_saved: List[LayerKey] = []
        for key in saved:
            if not key.name:
                continue
            group_key = key.group_name
            if isinstance(group_key, str):
                mapped = group_name_to_id.get((group_key, key.name))
                if mapped:
                    group_key = mapped
            if (group_key, key.name) not in current_set:
                candidates = current_by_name.get(key.name, [])
                if len(candidates) == 1:
                    group_key = candidates[0]
            normalized_saved.append(LayerKey(group_name=group_key, name=key.name))

        saved_filtered = []
        for key in normalized_saved:
            if self._key_tuple(key) in current_set:
                saved_filtered.append(key)
        saved_filtered = self._dedupe(saved_filtered)
        saved_set = {self._key_tuple(k) for k in saved_filtered}


        # Group order from saved order
        group_order: List[Optional[str]] = []
        for key in saved_filtered:
            if key.group_name not in group_order:
                group_order.append(key.group_name)

        # Track missing layers (in current order) by group
        missing_by_group: Dict[Optional[str], List[LayerKey]] = {}
        for key in current:
            key_tuple = self._key_tuple(key)
            if key_tuple in saved_set:
                continue
            missing_by_group.setdefault(key.group_name, []).append(key)

        # Add groups not present in saved order, in current order
        for key in current:
            if key.group_name not in group_order:
                group_order.append(key.group_name)

        # Build final order: saved keys (filtered) then missing by group
        saved_by_group: Dict[Optional[str], List[LayerKey]] = {}
        for key in saved_filtered:
            saved_by_group.setdefault(key.group_name, []).append(key)

        final: List[LayerKey] = []
        for group_id in group_order:
            final.extend(saved_by_group.get(group_id, []))
            final.extend(missing_by_group.get(group_id, []))

        if not final:
            final = current

        return final

    @staticmethod
    def _layers_to_keys(layers: Iterable[object]) -> List[LayerKey]:
        keys: List[LayerKey] = []
        sample = []
        for layer in layers:
            name = getattr(layer, "name", None)
            if not name:
                continue
            group_key = getattr(layer, "group_id", None) or getattr(layer, "group_name", None)
            keys.append(LayerKey(group_name=group_key, name=name))
            sample.append({
                "name": name,
                "group_name": getattr(layer, "group_name", None),
                "group_id": getattr(layer, "group_id", None),
            })
        return keys

    @staticmethod
    def _key_tuple(key: LayerKey) -> Tuple[Optional[str], str]:
        return (key.group_name, key.name)

    @staticmethod
    def _dedupe(keys: List[LayerKey]) -> List[LayerKey]:
        seen = set()
        deduped: List[LayerKey] = []
        for key in keys:
            key_tuple = (key.group_name, key.name)
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
            deduped.append(key)
        return deduped

    @staticmethod
    def _sync_prefix(group_name: Optional[str], layer_name: str, is_synced: bool) -> Optional[str]:
        if not is_synced or not group_name:
            return None
        if not group_name.startswith("TC "):
            return None
        tc_num = group_name.split(" ", 1)[1].strip()
        if not tc_num:
            return None
        return f"{tc_num}_"
