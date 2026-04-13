# GUI Input Demo Checklist

- [ ] Add runtime seams so Lane B can boot a deterministic timeline fixture through the same intent and presentation pipeline used by the UI
- [ ] Add a Lane B driver that uses real Qt mouse and keyboard simulation instead of direct state mutation
- [ ] Add a JSON DSL parser and validator with fail-fast errors for unsupported actions and missing required fields
- [ ] Add a starter scenario library under `tests/gui/scenarios/` with at least one realistic core flow
- [ ] Add `gui-lane-b` to the built-in lane runner so the flow is invokable from the repo CLI
- [ ] Add flake controls for offscreen GUI runs, deterministic fixture data, and stable artifact paths
- [ ] Add artifact publishing seams for trace JSON now, screenshots optionally, and video later
