"""
Multiple Instance Port Management for Toolathlon.

This module provides tools for managing port configurations when running
multiple Toolathlon instances on the same machine.

Main Components:
    - PortReplacer: Class for applying and restoring port changes
    - restore_all_ports: Function to restore all ports to defaults
    - Constants: DEFAULT_PORTS, INSTANCE_NAMES

Usage:
    from global_preparation.multiple_instance import (
        PortReplacer,
        restore_all_ports,
        DEFAULT_PORTS,
        parse_instance_id,
    )
"""

from .constants import DEFAULT_PORTS, INSTANCE_NAMES
from .utils import (
    parse_instance_id,
    calculate_port_offset,
    calculate_port_mappings,
    get_changes_log_name,
)
from .port_replacer import PortReplacer
from .restore import (
    find_all_changes_logs,
    check_if_ports_are_default,
    list_pending_changes,
    restore_all_ports,
    auto_detect_changes_log,
)

__all__ = [
    # Constants
    'DEFAULT_PORTS',
    'INSTANCE_NAMES',
    # Utils
    'parse_instance_id',
    'calculate_port_offset',
    'calculate_port_mappings',
    'get_changes_log_name',
    # PortReplacer
    'PortReplacer',
    # Restore functions
    'find_all_changes_logs',
    'check_if_ports_are_default',
    'list_pending_changes',
    'restore_all_ports',
    'auto_detect_changes_log',
]
