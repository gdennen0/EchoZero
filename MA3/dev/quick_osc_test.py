#!/usr/bin/env python3
"""
Quick OSC test - send one-liner commands to MA3

Usage:
    python quick_osc_test.py                    # Send default test
    python quick_osc_test.py "Printf('Hi')"    # Send custom Lua
    python quick_osc_test.py --raw "Go Exec 1" # Send raw MA3 command
"""

import socket
import struct
import sys

MA3_IP = "127.0.0.1"
MA3_PORT = 8000


def build_osc_string(s: str) -> bytes:
    """Build OSC string (null-padded to 4-byte boundary)"""
    encoded = s.encode('utf-8') + b'\x00'
    padding = (4 - len(encoded) % 4) % 4
    return encoded + b'\x00' * padding


def send_osc(address: str, *args):
    """Send OSC message"""
    # Build message
    msg = build_osc_string(address)
    
    # Type tag
    type_tag = ","
    for arg in args:
        if isinstance(arg, str):
            type_tag += "s"
        elif isinstance(arg, int):
            type_tag += "i"
        elif isinstance(arg, float):
            type_tag += "f"
    msg += build_osc_string(type_tag)
    
    # Arguments
    for arg in args:
        if isinstance(arg, str):
            msg += build_osc_string(arg)
        elif isinstance(arg, int):
            msg += struct.pack('>i', arg)
        elif isinstance(arg, float):
            msg += struct.pack('>f', arg)
    
    # Send
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(msg, (MA3_IP, MA3_PORT))
    sock.close()
    
    print(f"Sent to {MA3_IP}:{MA3_PORT}")
    print(f"  Address: {address}")
    print(f"  Args: {args}")


def main():
    if len(sys.argv) < 2:
        # Default test
        lua_code = "Printf('=== OSC TEST FROM ECHOZERO ===')"
        print("Sending default test message...")
    elif sys.argv[1] == "--raw":
        # Raw MA3 command (not wrapped in Lua)
        cmd = sys.argv[2] if len(sys.argv) > 2 else "Off Executor 1"
        send_osc("/cmd", cmd)
        return
    elif sys.argv[1] == "--help":
        print(__doc__)
        return
    else:
        lua_code = sys.argv[1]
    
    # Wrap in Lua command
    cmd = f'Lua "{lua_code}"'
    send_osc("/cmd", cmd)
    print(f"\nCheck MA3 System Monitor for: {lua_code}")


if __name__ == "__main__":
    main()
