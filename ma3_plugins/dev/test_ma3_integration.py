#!/usr/bin/env python3
"""
MA3 Integration Test Framework

Tests bidirectional OSC communication between EchoZero and grandMA3.
Sends commands and validates responses.

Usage:
    python3 test_ma3_integration.py              # Run all tests against live MA3
    python3 test_ma3_integration.py --mock       # Run with mock MA3 responses
    python3 test_ma3_integration.py --test ping  # Run specific test
    python3 test_ma3_integration.py --list       # List available tests
"""

import argparse
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable


# =============================================================================
# Configuration
# =============================================================================

MA3_IP = "127.0.0.1"
MA3_PORT = 8000      # MA3's OSC input port
LISTEN_PORT = 9000   # EchoZero's listening port for MA3 responses
TIMEOUT = 3.0        # Seconds to wait for response


# =============================================================================
# OSC Utilities
# =============================================================================

def osc_pad_string(s: str) -> bytes:
    """Pad string to 4-byte boundary (OSC requirement)."""
    encoded = s.encode('utf-8') + b'\x00'
    padding = (4 - len(encoded) % 4) % 4
    return encoded + b'\x00' * padding


def osc_pack_int32(n: int) -> bytes:
    """Pack 32-bit integer (big-endian)."""
    return struct.pack('>i', n)


def osc_pack_float32(f: float) -> bytes:
    """Pack 32-bit float (big-endian)."""
    return struct.pack('>f', f)


def build_osc_message(address: str, *args) -> bytes:
    """Build a complete OSC message."""
    msg = osc_pad_string(address)
    
    # Type tag
    type_tag = ","
    for arg in args:
        if isinstance(arg, str):
            type_tag += "s"
        elif isinstance(arg, int):
            type_tag += "i"
        elif isinstance(arg, float):
            type_tag += "f"
    msg += osc_pad_string(type_tag)
    
    # Arguments
    for arg in args:
        if isinstance(arg, str):
            msg += osc_pad_string(arg)
        elif isinstance(arg, int):
            msg += osc_pack_int32(arg)
        elif isinstance(arg, float):
            msg += osc_pack_float32(arg)
    
    return msg


def parse_osc_message(data: bytes) -> tuple:
    """Parse an OSC message, return (address, args)."""
    # Parse address
    end = data.index(b'\x00')
    address = data[:end].decode('utf-8')
    offset = end + 1
    offset += (4 - offset % 4) % 4  # Align to 4 bytes
    
    # Parse type tag
    if offset >= len(data):
        return address, []
    
    type_end = data.index(b'\x00', offset)
    type_tag = data[offset:type_end].decode('utf-8')
    offset = type_end + 1
    offset += (4 - offset % 4) % 4
    
    # Parse arguments based on type tag
    args = []
    for t in type_tag[1:]:  # Skip leading comma
        if t == 's':
            str_end = data.index(b'\x00', offset)
            args.append(data[offset:str_end].decode('utf-8'))
            offset = str_end + 1
            offset += (4 - offset % 4) % 4
        elif t == 'i':
            args.append(struct.unpack('>i', data[offset:offset+4])[0])
            offset += 4
        elif t == 'f':
            args.append(struct.unpack('>f', data[offset:offset+4])[0])
            offset += 4
        elif t in ('T', 'F', 'N', 'I'):
            # True, False, Nil, Impulse have no data
            args.append(t == 'T')
    
    return address, args


# =============================================================================
# Test Result
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str = ""
    response_address: str = ""
    response_args: List[Any] = field(default_factory=list)
    duration: float = 0.0


# =============================================================================
# Test Runner
# =============================================================================

class MA3TestRunner:
    """Runs tests against MA3 spine."""
    
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.results: List[TestResult] = []
        self._response: Optional[tuple] = None
        self._response_event = threading.Event()
        self._listener_sock: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        
    def start_listener(self):
        """Start listening for MA3 responses."""
        self._listener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listener_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener_sock.settimeout(0.5)
        self._listener_sock.bind(('0.0.0.0', LISTEN_PORT))
        self._running = True
        
        def listen():
            while self._running:
                try:
                    data, addr = self._listener_sock.recvfrom(4096)
                    address, args = parse_osc_message(data)
                    self._response = (address, args)
                    self._response_event.set()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        print(f"Listener error: {e}")
        
        self._listener_thread = threading.Thread(target=listen, daemon=True)
        self._listener_thread.start()
        
    def stop_listener(self):
        """Stop the listener."""
        self._running = False
        if self._listener_sock:
            self._listener_sock.close()
        if self._listener_thread:
            self._listener_thread.join(timeout=1.0)
    
    def send_lua(self, lua_code: str) -> bool:
        """Send a Lua command to MA3 via OSC /cmd."""
        msg = build_osc_message("/cmd", f'Lua "{lua_code}"')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(msg, (MA3_IP, MA3_PORT))
        sock.close()
        return True
    
    def wait_for_response(self, expected_address: str, timeout: float = TIMEOUT) -> Optional[tuple]:
        """Wait for a response with the expected address."""
        self._response = None
        self._response_event.clear()
        
        start = time.time()
        while time.time() - start < timeout:
            if self._response_event.wait(timeout=0.1):
                if self._response and self._response[0] == expected_address:
                    return self._response
                self._response_event.clear()
        
        return None
    
    def run_test(self, name: str, lua_code: str, expected_address: str, 
                 validator: Optional[Callable[[List[Any]], bool]] = None) -> TestResult:
        """Run a single test."""
        start = time.time()
        
        if self.mock_mode:
            # Simulate response
            result = TestResult(
                name=name,
                passed=True,
                message="Mock mode - simulated success",
                response_address=expected_address,
                response_args=["mock_response"],
                duration=0.01
            )
            self.results.append(result)
            return result
        
        # Send command
        self.send_lua(lua_code)
        
        # Wait for response
        response = self.wait_for_response(expected_address)
        duration = time.time() - start
        
        if response is None:
            result = TestResult(
                name=name,
                passed=False,
                message=f"Timeout waiting for {expected_address}",
                duration=duration
            )
        else:
            address, args = response
            passed = True
            message = "OK"
            
            if validator:
                try:
                    passed = validator(args)
                    if not passed:
                        message = f"Validation failed: {args}"
                except Exception as e:
                    passed = False
                    message = f"Validator error: {e}"
            
            result = TestResult(
                name=name,
                passed=passed,
                message=message,
                response_address=address,
                response_args=args,
                duration=duration
            )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: TestResult):
        """Print a test result."""
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name} ({result.duration:.3f}s)")
        if not result.passed:
            print(f"         {result.message}")
        elif result.response_args:
            args_str = str(result.response_args)
            if len(args_str) > 60:
                args_str = args_str[:60] + "..."
            print(f"         Response: {result.response_address} {args_str}")
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print()
        print("=" * 50)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 50)


# =============================================================================
# Test Definitions
# =============================================================================

def define_tests() -> Dict[str, dict]:
    """Define all available tests."""
    return {
        "ping": {
            "description": "Test basic connectivity with ping/pong",
            "lua_code": "EZ.Ping()",
            "expected_address": "/ma3/pong",
            "validator": lambda args: len(args) > 0 and isinstance(args[0], int)
        },
        "echo": {
            "description": "Test echo response",
            "lua_code": "EZ.Echo('hello_echozero')",
            "expected_address": "/ma3/echo",
            "validator": lambda args: len(args) > 0 and args[0] == "hello_echozero"
        },
        "status": {
            "description": "Test status report",
            "lua_code": "EZ.Status()",
            "expected_address": "/ma3/status",
            "validator": lambda args: len(args) > 0 and "v2.0.0" in str(args[0])
        },
        "list_trackgroups": {
            "description": "List track groups in timecode 1",
            "lua_code": "EZ.ListTrackGroups(1)",
            "expected_address": "/ma3/track_groups",
            "validator": None  # Just check we get a response
        },
        "list_tracks": {
            "description": "List tracks in timecode 1, trackgroup 1",
            "lua_code": "EZ.ListTracks(1, 1)",
            "expected_address": "/ma3/tracks",
            "validator": None
        },
        "get_structure": {
            "description": "Get full timecode structure",
            "lua_code": "EZ.GetStructure(1)",
            "expected_address": "/ma3/structure",
            "validator": None
        },
        "add_event": {
            "description": "Add test event at 5 seconds",
            "lua_code": "EZ.AddEvent(1, 1, 1, 5.0, 'test_kick', 'cmd')",
            "expected_address": "/ma3/event_added",
            "validator": lambda args: len(args) >= 4
        },
    }


# =============================================================================
# Main
# =============================================================================

def main():
    global MA3_IP, MA3_PORT
    
    parser = argparse.ArgumentParser(description="MA3 Integration Tests")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no MA3 needed)")
    parser.add_argument("--test", type=str, help="Run specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--port", type=int, default=8000, help="MA3 OSC port")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="MA3 IP address")
    args = parser.parse_args()
    
    MA3_IP = args.ip
    MA3_PORT = args.port
    
    tests = define_tests()
    
    if args.list:
        print("Available tests:")
        for name, info in tests.items():
            print(f"  {name}: {info['description']}")
        return
    
    print("=" * 50)
    print("MA3 Integration Tests")
    print("=" * 50)
    print(f"MA3 Target: {MA3_IP}:{MA3_PORT}")
    print(f"Listen Port: {LISTEN_PORT}")
    print(f"Mock Mode: {args.mock}")
    print()
    
    runner = MA3TestRunner(mock_mode=args.mock)
    
    if not args.mock:
        print("Starting listener...")
        runner.start_listener()
        time.sleep(0.5)  # Give listener time to start
    
    try:
        if args.test:
            # Run specific test
            if args.test not in tests:
                print(f"Unknown test: {args.test}")
                print("Use --list to see available tests")
                return
            
            test_info = tests[args.test]
            print(f"Running test: {args.test}")
            result = runner.run_test(
                name=args.test,
                lua_code=test_info["lua_code"],
                expected_address=test_info["expected_address"],
                validator=test_info.get("validator")
            )
            runner.print_result(result)
        else:
            # Run all tests
            print("Running all tests...")
            print()
            
            for name, info in tests.items():
                result = runner.run_test(
                    name=name,
                    lua_code=info["lua_code"],
                    expected_address=info["expected_address"],
                    validator=info.get("validator")
                )
                runner.print_result(result)
        
        runner.print_summary()
        
    finally:
        if not args.mock:
            runner.stop_listener()


if __name__ == "__main__":
    main()
