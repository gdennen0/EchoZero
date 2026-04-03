from echozero.application.presentation.models import TakeActionPresentation, TakeLanePresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import TakeId
from echozero.ui.qt.timeline.blocks.layouts import TakeRowLayout
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock


def _take(actions=None) -> TakeLanePresentation:
    return TakeLanePresentation(
        take_id=TakeId("take_test"),
        name="Take 2",
        kind=LayerKind.EVENT,
        actions=actions or [],
    )


def test_take_row_default_actions_include_overwrite_and_merge():
    block = TakeRowBlock()
    actions = block._actions_for_take(_take())
    action_ids = {action.action_id for action in actions}

    assert {"overwrite_main", "merge_main"} <= action_ids


def test_take_row_uses_explicit_take_actions_when_provided():
    block = TakeRowBlock()
    custom = [TakeActionPresentation(action_id="custom_action", label="Custom")]
    actions = block._actions_for_take(_take(actions=custom))

    assert len(actions) == 1
    assert actions[0].action_id == "custom_action"


def test_take_row_layout_reserves_options_area():
    layout = TakeRowLayout.create(top=10, width=900, header_width=320, row_height=44)

    assert layout.options_button_rect.width() > 0
    assert layout.options_area_rect.width() > 0
    assert layout.options_area_rect.top() > layout.label_rect.top()
