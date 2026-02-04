#!/usr/bin/env python3
"""
Apply port number changes across the Toolathlon codebase.

This script automatically manages port configurations for running multiple
Toolathlon instances on the same machine without port conflicts.

FEATURES:
    - Supports parallel instances: alpha and beta can coexist
    - Auto-detects current port values in files (even if modified by other instances)
    - Supports both automatic offset and manual port configuration
    - Always generates change log for tracking (required for restore)
    - Unified restore: scans all change logs and restores all ports at once

Usage:
    # Apply port changes using config file (default behavior)
    python global_preparation/apply_port_numbers.py
    python global_preparation/apply_port_numbers.py --dry-run

    # Apply port changes for instance 1 (or 'alpha') - auto-calculated ports
    python global_preparation/apply_port_numbers.py --instance 1
    python global_preparation/apply_port_numbers.py --instance alpha

    # Apply port changes for instance 2 (or 'beta')  
    python global_preparation/apply_port_numbers.py --instance 2
    python global_preparation/apply_port_numbers.py --instance beta

    # Check current port status (default or modified)
    python global_preparation/apply_port_numbers.py --status

    # Restore all ports to defaults (scans all change logs)
    python global_preparation/apply_port_numbers.py --restore

    # List all pending port changes
    python global_preparation/apply_port_numbers.py --list

Port Mapping Rules:
    Instance 1 (alpha): offset +1000  -> 10001 becomes 11001
    Instance 2 (beta):  offset +2000  -> 10001 becomes 12001
    Instance 3 (gamma): offset +3000  -> 10001 becomes 13001
    ...and so on

Instance Names:
    1, 2, 3, ...  or  alpha, beta, gamma, delta, epsilon, ...

Restore Behavior:
    - Reads ALL port_changes_*.json files
    - Collects all modified port mappings (e.g., 10001 -> [11001, 12001])
    - Scans ALL files in files_by_port
    - Replaces ANY modified port with its default value
    - Deletes all change log files after successful restore

Module Structure:
    This script uses modules from global_preparation/multiple_instance/:
    - constants.py: DEFAULT_PORTS, INSTANCE_NAMES
    - utils.py: parse_instance_id, calculate_port_offset, etc.
    - port_replacer.py: PortReplacer class
    - restore.py: restore_all_ports, list_pending_changes, etc.
"""

import argparse
import sys
from pathlib import Path

# Import from the multiple_instance module
from multiple_instance import (
    PortReplacer,
    parse_instance_id,
    get_changes_log_name,
    find_all_changes_logs,
    check_if_ports_are_default,
    list_pending_changes,
    restore_all_ports,
)


def main():
    """Main entry point for the port management script."""
    parser = argparse.ArgumentParser(
        description='Apply or restore port number changes for multi-instance Toolathlon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply using port_mappings from config file (default)
  python global_preparation/apply_port_numbers.py
  python global_preparation/apply_port_numbers.py --dry-run

  # Apply with auto-calculated ports for instance 1 (alpha)
  python global_preparation/apply_port_numbers.py --instance 1
  python global_preparation/apply_port_numbers.py --instance alpha

  # Apply for instance 2 (beta)
  python global_preparation/apply_port_numbers.py --instance 2

  # List all pending port changes
  python global_preparation/apply_port_numbers.py --list

  # Restore all instances
  python global_preparation/apply_port_numbers.py --restore

Default Behavior:
  Uses port_mappings from configs/ports_config.yaml

Auto-calculated Ports (--instance):
  Instance 1 (alpha): 10001 -> 11001, 20001 -> 21001, ...
  Instance 2 (beta):  10001 -> 12001, 20001 -> 22001, ...
  Instance N:         port -> port + N * 1000
        """
    )

    parser.add_argument(
        '--instance', '-i',
        help='Instance identifier (1, 2, 3... or alpha, beta, gamma...)'
    )

    parser.add_argument(
        '--config',
        default='configs/ports_config.yaml',
        help='Path to ports config file (default: configs/ports_config.yaml)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would change without modifying files'
    )

    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore original port numbers'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='(Deprecated) --restore now always restores all ports'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all pending port changes'
    )

    parser.add_argument(
        '--use-git',
        action='store_true',
        help='When used with --restore, use git restore (WARNING: restores ALL files)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check current port status (default or modified)'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Validate arguments
    if args.use_git and not args.restore:
        print("Error: --use-git can only be used with --restore")
        sys.exit(1)
    
    if args.all and not args.restore:
        print("Error: --all can only be used with --restore")
        sys.exit(1)

    # Handle --list
    if args.list:
        list_pending_changes(project_root)
        sys.exit(0)

    # Handle --status
    if args.status:
        _handle_status(project_root, args.config)
        sys.exit(0)

    # Handle --restore
    if args.restore:
        _handle_restore(project_root, args.config, args.dry_run, args.use_git)
        sys.exit(0)

    # Handle --instance (apply changes with auto-calculated ports)
    if args.instance:
        _handle_apply(project_root, args.instance, args.config, args.dry_run)
        sys.exit(0)

    # Default action: apply changes using config file's port_mappings
    _handle_apply_from_config(project_root, args.config, args.dry_run)
    sys.exit(0)


def _handle_status(project_root: Path, config_path: str) -> None:
    """Handle --status command."""
    print("Checking current port status...")
    print("=" * 70)
    
    # Check for pending changes logs
    changes_files = find_all_changes_logs(project_root)
    if changes_files:
        print(f"\n📄 Pending changes files: {len(changes_files)}")
        for f in changes_files:
            print(f"   - {f.name}")
    else:
        print("\n📄 No pending changes files found")
    
    # Check actual port values in code
    is_default, detected_mappings = check_if_ports_are_default(project_root, config_path)
    
    print(f"\n🔍 Port detection results:")
    if is_default:
        print("   Status: ✓ All ports are at DEFAULT values")
    else:
        print("   Status: ⚠ Ports have been MODIFIED")
    
    print(f"\n   Detected port mappings:")
    for default_port, current_port in sorted(detected_mappings.items()):
        if default_port == current_port:
            print(f"     {default_port}: {current_port} (default)")
        else:
            print(f"     {default_port}: {current_port} (modified, offset +{current_port - default_port})")
    
    print()
    print("=" * 70)
    
    if not is_default and not changes_files:
        print("\n⚠ Warning: Ports are modified but no changes log found!")
        print("   This may happen if you manually modified ports.")
        print("   To create a changes log for restoration, run:")
        print(f"     python {__file__} --instance <id>")
        print("   Or use git restore to reset all files.")


def _handle_restore(project_root: Path, config_path: str, dry_run: bool, use_git: bool) -> None:
    """Handle --restore command."""
    if use_git:
        replacer = PortReplacer(config_path, dry_run=dry_run)
        success = replacer.restore_changes(use_git=True)
        sys.exit(0 if success else 1)
    
    # Use unified restore function that handles all change logs
    success = restore_all_ports(project_root, config_path, dry_run=dry_run)
    sys.exit(0 if success else 1)


def _handle_apply(project_root: Path, instance: str, config_path: str, dry_run: bool) -> None:
    """Handle --instance command (apply changes with auto-calculated ports)."""
    instance_id, instance_name = parse_instance_id(instance)
    
    # Check if THIS instance already has pending changes (prevent duplicate apply)
    expected_log = project_root / get_changes_log_name(instance_name)
    if expected_log.exists() and not dry_run:
        print(f"Warning: Instance '{instance_name}' already has pending changes.")
        print(f"  Changes file: {expected_log}")
        print()
        print("Options:")
        print(f"  1. Restore first: python {__file__} --restore")
        print(f"  2. View details:  python {__file__} --list")
        sys.exit(1)
    
    # Apply changes
    replacer = PortReplacer(
        config_path, 
        dry_run=dry_run,
        instance_id=instance_id,
        instance_name=instance_name
    )
    success = replacer.apply_changes()
    sys.exit(0 if success or dry_run else 1)


def _handle_apply_from_config(project_root: Path, config_path: str, dry_run: bool) -> None:
    """Handle --apply command (apply changes using port_mappings from config file)."""
    # Check if there's already a pending change log for manual config
    expected_log = project_root / "port_changes_manual.json"
    if expected_log.exists() and not dry_run:
        print("Warning: Manual configuration already has pending changes.")
        print(f"  Changes file: {expected_log}")
        print()
        print("Options:")
        print(f"  1. Restore first: python {__file__} --restore")
        print(f"  2. View details:  python {__file__} --list")
        sys.exit(1)
    
    # Apply changes using config file's port_mappings
    # instance_name='manual' ensures change log is named port_changes_manual.json
    replacer = PortReplacer(
        config_path, 
        dry_run=dry_run,
        instance_id=None,  # Don't auto-calculate, use config
        instance_name='manual'  # For change log naming
    )
    success = replacer.apply_changes()
    sys.exit(0 if success or dry_run else 1)


if __name__ == '__main__':
    main()
