| Priority | Task | Status |
| --- | --- | --- |
| P1 | Extract Foundry UI read queries from `FoundryApp` into a dedicated query service and keep app query APIs delegating for compatibility. | DONE |
| P2 | Move Foundry UI write-side actions behind explicit command/service boundaries so the UI stops coordinating domain mutations directly. | TODO |
| P3 | Isolate Foundry UI view-model assembly from Qt widgets so screen refresh logic becomes testable without widget setup. | TODO |
| P4 | Tighten Foundry smoke coverage around query and command boundaries to catch composition-root regressions early. | TODO |
