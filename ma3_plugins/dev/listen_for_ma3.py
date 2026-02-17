#!/usr/bin/env python3
"""
Listen for OSC messages from MA3

Run this, then in MA3:
    1. First load the spine: Lua "dofile('plugins/echozero_spine/init.lua')"
    2. Test with: Lua "EZ.Test()"
    
    Or manually:
    Cmd("SendOSC EchoZero \"/test,s,hello\"")

Press Ctrl+C to stop.
"""

import socket
import sys

LISTEN_PORT = 9000


def parse_osc_string(data: bytes, offset: int) -> tuple:
    """Parse null-terminated OSC string"""
    end = data.index(b'\x00', offset)
    s = data[offset:end].decode('utf-8')
    # Align to 4 bytes
    padding = (4 - (end - offset + 1) % 4) % 4
    return s, end + 1 + padding


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN_PORT
    
    print("=" * 50)
    print(f"Listening for OSC on port {port}")
    print("=" * 50)
    print()
    print("In MA3, run:")
    print(f'  Lua "SendOSC(\'EchoZero\', \'/test\', \'hello\')"')
    print()
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            
            # Parse OSC address
            try:
                address, offset = parse_osc_string(data, 0)
                print(f"\n[RECEIVED] from {addr[0]}:{addr[1]}")
                print(f"  Address: {address}")
                print(f"  Raw data ({len(data)} bytes): {data[:64]}...")
                
                # Try to parse type tag and args
                if offset < len(data):
                    type_tag, offset = parse_osc_string(data, offset)
                    print(f"  Type tag: {type_tag}")
                    
            except Exception as e:
                print(f"\n[RECEIVED] from {addr[0]}:{addr[1]}")
                print(f"  Raw: {data}")
                print(f"  Parse error: {e}")
                
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
