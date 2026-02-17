"""
Unit tests for ShowManager interface discovery using netifaces.
"""

import netifaces


def test_netifaces_lists_interfaces():
    """netifaces should return interface names on macOS."""
    interfaces = netifaces.interfaces()
    assert isinstance(interfaces, list)
    assert len(interfaces) > 0
    # macOS always has lo0 (loopback)
    assert "lo0" in interfaces
    print(f"Interfaces found: {interfaces}")


def test_netifaces_gets_ipv4_addresses():
    """netifaces should return IPv4 addresses for interfaces."""
    found_ipv4 = []
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        ipv4_list = addrs.get(netifaces.AF_INET, [])
        for entry in ipv4_list:
            ip = entry.get('addr')
            if ip:
                found_ipv4.append((iface, ip))
    
    assert len(found_ipv4) > 0, "Should find at least one IPv4 address"
    # lo0 should have 127.0.0.1
    lo0_ips = [ip for iface, ip in found_ipv4 if iface == "lo0"]
    assert "127.0.0.1" in lo0_ips, "lo0 should have 127.0.0.1"
    print(f"IPv4 addresses found: {found_ipv4}")


def test_interface_candidates_include_loopback_and_all():
    """Interface discovery should always include loopback and all interfaces."""
    from ui.qt_gui.block_panels.show_manager_panel import ShowManagerPanel
    panel = ShowManagerPanel.__new__(ShowManagerPanel)
    candidates = panel._get_interface_candidates()
    assert isinstance(candidates, list)
    assert ("All Interfaces", "0.0.0.0") in candidates
    assert ("Loopback", "127.0.0.1") in candidates
    print(f"Candidates from panel: {candidates}")
def test_interface_discovery_finds_real_interfaces():
    """Interface discovery should find at least one real network interface beyond loopback."""
    from ui.qt_gui.block_panels.show_manager_panel import ShowManagerPanel
    panel = ShowManagerPanel.__new__(ShowManagerPanel)
    candidates = panel._get_interface_candidates()
    
    # Filter out the hardcoded entries
    real_interfaces = [
        (name, addr) for name, addr in candidates 
        if addr not in ("0.0.0.0", "127.0.0.1")
    ]
    
    # On any real machine with networking, should have at least one interface
    assert len(real_interfaces) >= 1, "Should find at least one real network interface"
    print(f"Real interfaces found: {real_interfaces}")