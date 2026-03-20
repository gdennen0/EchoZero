"""
UI package: PySide6 desktop interface for EchoZero (D245).
Separate process from Core engine. Communicates via HTTP + WebSocket API (D244).
No domain logic here — UI is a pure API client. Engine Ignorance (FP7) applies in reverse.
"""
