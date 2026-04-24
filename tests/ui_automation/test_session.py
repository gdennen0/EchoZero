from __future__ import annotations

from ui_automation import (
    AutomationAction,
    AutomationBounds,
    AutomationHitTarget,
    AutomationObject,
    AutomationObjectFact,
    AutomationSession,
    AutomationSnapshot,
    AutomationTarget,
)


class _StubBackend:
    def __init__(self) -> None:
        self.closed = False
        self.last_clicked: str | None = None
        self.last_pointer_target: str | None = None
        self.last_double_clicked: str | None = None
        self.last_hovered: str | None = None
        self.last_typed: tuple[str, str] | None = None
        self.last_key: str | None = None
        self.last_drag: tuple[str, object] | None = None
        self.last_scroll: tuple[str, int, int] | None = None

    def snapshot(self) -> AutomationSnapshot:
        return AutomationSnapshot(
            app="EchoZero",
            selection=("shell.timeline",),
            focused_target_id="shell.timeline",
            targets=(
                AutomationTarget(
                    kind="canvas",
                    target_id="shell.timeline",
                    label="Timeline",
                    bounds=AutomationBounds(x=10, y=20, width=300, height=200),
                ),
                AutomationTarget(
                    kind="toolbar",
                    target_id="shell.transport",
                    label="Transport",
                ),
            ),
            actions=(
                AutomationAction(
                    action_id="timeline.zoom_in",
                    label="Zoom In",
                    group="timeline",
                    target_id="shell.timeline",
                ),
            ),
            objects=(
                AutomationObject(
                    object_id="source_audio",
                    object_type="layer",
                    label="Source Audio",
                    target_id="timeline.layer:source_audio",
                    facts=(AutomationObjectFact(label="kind", value="source_audio"),),
                    actions=(
                        AutomationAction(
                            action_id="timeline.extract_stems",
                            label="Extract Stems",
                            group="timeline",
                            target_id="timeline.layer:source_audio",
                        ),
                    ),
                ),
                AutomationObject(
                    object_id="event_kick_1",
                    object_type="event",
                    label="Kick 1",
                    target_id="timeline.event:event_kick_1",
                    facts=(AutomationObjectFact(label="classification", value="kick"),),
                ),
            ),
            hit_targets=(
                AutomationHitTarget(
                    target_id="timeline.layer:source_audio",
                    kind="layer",
                    bounds=AutomationBounds(x=10, y=20, width=300, height=40),
                ),
                AutomationHitTarget(
                    target_id="timeline.event:event_kick_1",
                    kind="event",
                    bounds=AutomationBounds(x=24, y=72, width=12, height=12),
                ),
            ),
        )

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        if target_id is None:
            return b"window-shot"
        return f"shot:{target_id}".encode("utf-8")

    def click(
        self,
        target_id: str,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_clicked = target_id
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def move_pointer(
        self,
        target_id: str,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_pointer_target = target_id
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def double_click(
        self,
        target_id: str,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_double_clicked = target_id
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def hover(
        self,
        target_id: str,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_hovered = target_id
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def type_text(
        self,
        target_id: str,
        text: str,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_typed = (target_id, text)
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def press_key(
        self,
        key: str,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_key = key
        return AutomationSnapshot(app="EchoZero")

    def drag(
        self,
        target_id: str,
        destination: object,
        *,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_drag = (target_id, destination)
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def scroll(
        self,
        target_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
        args: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        self.last_scroll = (target_id, dx, dy)
        return AutomationSnapshot(app="EchoZero", selection=(target_id,))

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, object] | None = None,
    ) -> AutomationSnapshot:
        return AutomationSnapshot(
            app="EchoZero",
            selection=(target_id,) if target_id is not None else (),
            actions=(AutomationAction(action_id=action_id, label=action_id, params=params or {}),),
        )

    def close(self) -> None:
        self.closed = True


def test_session_returns_snapshot_and_screenshot():
    session = AutomationSession.attach(_StubBackend())

    snapshot = session.snapshot()

    assert snapshot.app == "EchoZero"
    assert snapshot.selection == ("shell.timeline",)
    assert snapshot.focused_target_id == "shell.timeline"
    assert snapshot.objects[0].object_type == "layer"
    assert session.screenshot(target_id="shell.timeline") == b"shot:shell.timeline"


def test_session_finds_targets_by_exact_or_partial_query():
    session = AutomationSession.attach(_StubBackend())

    assert session.find_target("shell.timeline") is not None
    assert session.find_target("transport").target_id == "shell.transport"  # type: ignore[union-attr]
    assert session.find_target("missing") is None


def test_session_finds_objects_by_exact_partial_and_filtered_query():
    session = AutomationSession.attach(_StubBackend())

    assert session.find_object("source_audio").object_id == "source_audio"  # type: ignore[union-attr]
    assert session.find_object("Kick").object_id == "event_kick_1"  # type: ignore[union-attr]
    assert session.find_object("classification", object_type="event").object_id == "event_kick_1"  # type: ignore[union-attr]
    assert session.find_object("source", object_type="event") is None


def test_session_finds_actions_across_object_and_snapshot_surfaces():
    session = AutomationSession.attach(_StubBackend())

    assert session.find_action("timeline.zoom_in").action_id == "timeline.zoom_in"  # type: ignore[union-attr]
    assert session.find_action("extract", target_id="timeline.layer:source_audio").action_id == "timeline.extract_stems"  # type: ignore[union-attr]
    assert session.find_action("extract", group="transport") is None


def test_session_finds_hit_targets_by_exact_partial_and_kind():
    session = AutomationSession.attach(_StubBackend())

    assert session.find_hit_target("timeline.layer:source_audio").kind == "layer"  # type: ignore[union-attr]
    assert session.find_hit_target("kick", kind="event").target_id == "timeline.event:event_kick_1"  # type: ignore[union-attr]
    assert session.find_hit_target("source", kind="event") is None


def test_session_invokes_semantic_action_through_backend():
    session = AutomationSession.attach(_StubBackend())

    snapshot = session.invoke(
        "timeline.zoom_in",
        target_id="shell.timeline",
        params={"delta": 1},
    )

    assert snapshot.actions[0].action_id == "timeline.zoom_in"
    assert snapshot.selection == ("shell.timeline",)


def test_session_supports_pointer_click_type_key_drag_and_scroll():
    backend = _StubBackend()
    session = AutomationSession.attach(backend)

    pointer_snapshot = session.move_pointer("shell.timeline")
    click_snapshot = session.click("shell.timeline")
    double_click_snapshot = session.double_click("shell.timeline")
    hover_snapshot = session.hover("shell.timeline")
    typed_snapshot = session.type_text("shell.timeline", "Kick")
    key_snapshot = session.press_key("space")
    drag_snapshot = session.drag("shell.timeline", {"dx": 120, "dy": 0})
    scroll_snapshot = session.scroll("shell.timeline", dx=240, dy=0)

    assert backend.last_pointer_target == "shell.timeline"
    assert backend.last_clicked == "shell.timeline"
    assert backend.last_double_clicked == "shell.timeline"
    assert backend.last_hovered == "shell.timeline"
    assert backend.last_typed == ("shell.timeline", "Kick")
    assert backend.last_key == "space"
    assert backend.last_drag == ("shell.timeline", {"dx": 120, "dy": 0})
    assert backend.last_scroll == ("shell.timeline", 240, 0)
    assert pointer_snapshot.selection == ("shell.timeline",)
    assert click_snapshot.selection == ("shell.timeline",)
    assert double_click_snapshot.selection == ("shell.timeline",)
    assert hover_snapshot.selection == ("shell.timeline",)
    assert typed_snapshot.selection == ("shell.timeline",)
    assert key_snapshot.app == "EchoZero"
    assert drag_snapshot.selection == ("shell.timeline",)
    assert scroll_snapshot.selection == ("shell.timeline",)


def test_session_close_delegates_to_provider():
    backend = _StubBackend()
    session = AutomationSession.attach(backend)

    session.close()

    assert backend.closed is True
