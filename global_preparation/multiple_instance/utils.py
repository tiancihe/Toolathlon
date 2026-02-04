"""
Utility functions for multi-instance port management.
"""

import sys
from typing import Dict, List, Tuple

from .constants import DEFAULT_PORTS, INSTANCE_NAMES


def parse_instance_id(instance: str) -> Tuple[int, str]:
    """
    Parse instance identifier to get numeric ID and name.
    
    Args:
        instance: Either a number (1, 2, 3...) or a name (alpha, beta, gamma...)
    
    Returns:
        Tuple of (numeric_id, instance_name)
    """
    instance_lower = instance.lower().strip()
    
    # Try to parse as number
    try:
        num = int(instance_lower)
        if num < 1:
            print(f"Error: Instance number must be >= 1, got {num}")
            sys.exit(1)
        # Convert number to name if possible
        reverse_names = {v: k for k, v in INSTANCE_NAMES.items()}
        name = reverse_names.get(num, f"inst{num}")
        return num, name
    except ValueError:
        pass
    
    # Try to parse as name
    if instance_lower in INSTANCE_NAMES:
        return INSTANCE_NAMES[instance_lower], instance_lower
    
    print(f"Error: Unknown instance '{instance}'")
    print(f"Valid names: {', '.join(INSTANCE_NAMES.keys())}")
    print("Or use a number: 1, 2, 3, ...")
    sys.exit(1)


def calculate_port_offset(instance_id: int) -> int:
    """Calculate port offset based on instance ID."""
    return instance_id * 1000


def calculate_port_mappings(instance_id: int, default_ports: List[int] = None) -> Dict[int, int]:
    """
    Calculate port mappings for a given instance.
    
    Args:
        instance_id: Numeric instance ID (1, 2, 3, ...)
        default_ports: List of default ports (uses DEFAULT_PORTS if not provided)
    
    Returns:
        Dictionary mapping old_port -> new_port
    """
    if default_ports is None:
        default_ports = DEFAULT_PORTS
    offset = calculate_port_offset(instance_id)
    return {port: port + offset for port in default_ports}


def get_changes_log_name(instance_name: str) -> str:
    """Generate changes log filename based on instance name."""
    return f"port_changes_{instance_name}.json"
