"""
Functions for restoring ports to default values and managing change logs.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    import sys
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

from .constants import DEFAULT_PORTS


def find_all_changes_logs(project_root: Path) -> List[Path]:
    """Find all port_changes*.json files in the project root."""
    return sorted(project_root.glob("port_changes*.json"))


def check_if_ports_are_default(project_root: Path, config_path: str) -> Tuple[bool, Dict[int, int]]:
    """
    Check if the codebase currently has default ports or modified ports.
    
    Returns:
        Tuple of (is_default, detected_mappings)
        - is_default: True if all ports are at default values
        - detected_mappings: Dict of {default_port: current_port} detected in files
    """
    config_file = project_root / config_path
    if not config_file.exists():
        return True, {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    files_by_port = config.get('files_by_port', {})
    detected_mappings = {}
    
    # For each default port, check if it exists in files or if a modified version exists
    for default_port in DEFAULT_PORTS:
        default_port_str = str(default_port)
        port_info = files_by_port.get(default_port, files_by_port.get(str(default_port), {}))
        files = port_info.get('files', [])
        
        if not files:
            detected_mappings[default_port] = default_port
            continue
        
        # Check the first file that should contain this port
        sample_file = project_root / files[0]
        if not sample_file.exists():
            detected_mappings[default_port] = default_port
            continue
        
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            detected_mappings[default_port] = default_port
            continue
        
        # Use regex to match standalone port numbers (not part of larger numbers)
        def port_exists(port: int) -> bool:
            pattern = rf'(?<![0-9]){port}(?![0-9])'
            return bool(re.search(pattern, content))
        
        # Check if default port exists
        if port_exists(default_port):
            detected_mappings[default_port] = default_port
        else:
            # Try to find modified port (check offsets 1000-10000)
            found_port = None
            for offset in range(1000, 11000, 1000):
                modified_port = default_port + offset
                if port_exists(modified_port):
                    found_port = modified_port
                    break
            
            if found_port:
                detected_mappings[default_port] = found_port
            else:
                # Port not found, assume default
                detected_mappings[default_port] = default_port
    
    # Check if all ports are at default values
    is_default = all(default == current for default, current in detected_mappings.items())
    
    return is_default, detected_mappings


def list_pending_changes(project_root: Path) -> None:
    """List all pending port changes that can be restored."""
    changes_files = find_all_changes_logs(project_root)
    
    if not changes_files:
        print("No pending port changes found.")
        print()
        print("To apply port changes for a new instance:")
        print("  python global_preparation/apply_port_numbers.py --instance 1")
        print("  python global_preparation/apply_port_numbers.py --instance alpha")
        return
    
    print(f"Found {len(changes_files)} pending port change(s):")
    print("=" * 70)
    
    for changes_file in changes_files:
        try:
            with open(changes_file, 'r') as f:
                log_data = json.load(f)
            
            instance_name = log_data.get('instance_name', 'unknown')
            instance_id = log_data.get('instance_id', 'unknown')
            applied_at = log_data.get('applied_at', 'unknown')
            num_changes = len(log_data.get('changes', []))
            port_offset = log_data.get('port_offset', 'unknown')
            
            # Get unique port mappings
            port_mappings = {}
            for change in log_data.get('changes', []):
                old_port = change.get('default_port', change.get('old_port'))
                new_port = change['new_port']
                port_mappings[old_port] = new_port
            
            print(f"\n📄 {changes_file.name}")
            print(f"   Instance: {instance_name} (ID: {instance_id})")
            print(f"   Port offset: +{port_offset}")
            print(f"   Applied: {applied_at}")
            print(f"   Changes: {num_changes} file modifications")
            print(f"   Ports: {', '.join(f'{k}→{v}' for k, v in sorted(port_mappings.items())[:5])}...")
            
        except Exception as e:
            print(f"\n📄 {changes_file.name}")
            print(f"   Error reading file: {e}")
    
    print()
    print("=" * 70)
    print("\nTo restore:")
    print("  python global_preparation/apply_port_numbers.py --restore")


def restore_all_ports(project_root: Path, config_path: str = 'configs/ports_config.yaml', 
                      dry_run: bool = False) -> bool:
    """
    Restore all ports to default values by scanning all change logs.
    
    This function:
    1. Reads all port_changes_*.json files
    2. Collects all new_port -> default_port mappings
    3. For each file in files_by_port, finds any modified port and restores to default
    4. Deletes all change log files after successful restoration
    
    IMPORTANT: Only ports recorded in change logs will be restored.
    
    Args:
        project_root: Path to project root
        config_path: Path to ports config file
        dry_run: If True, only show what would be done
    
    Returns:
        True if successful, False otherwise
    """
    print(f"{'[DRY RUN] ' if dry_run else ''}Restoring all ports to default values...")
    print("=" * 70)
    
    # Step 1: Find all change logs
    changes_files = find_all_changes_logs(project_root)
    
    if not changes_files:
        print("\n⚠ No pending port changes found (no change log files).")
        print("   Nothing to restore.")
        print()
        print("   If ports were modified without using this script,")
        print("   please use: git restore . to reset all files.")
        return True
    
    print(f"\n📄 Found {len(changes_files)} change log(s):")
    for f in changes_files:
        print(f"   - {f.name}")
    
    # Step 2: Collect all port mappings and affected files from change logs
    port_restore_map: Dict[int, set] = {}
    files_to_restore: Dict[str, set] = {}  # file -> set of (default_port, new_port) tuples
    
    for changes_file in changes_files:
        try:
            with open(changes_file, 'r') as f:
                log_data = json.load(f)
            
            for change in log_data.get('changes', []):
                default_port = change.get('default_port', change.get('old_port'))
                new_port = change['new_port']
                file_path = change.get('file')
                
                if default_port not in port_restore_map:
                    port_restore_map[default_port] = set()
                port_restore_map[default_port].add(new_port)
                
                # Track which files need restoration
                if file_path:
                    if file_path not in files_to_restore:
                        files_to_restore[file_path] = set()
                    files_to_restore[file_path].add((default_port, new_port))
                
        except Exception as e:
            print(f"   ⚠ Error reading {changes_file.name}: {e}")
    
    if not port_restore_map:
        print("\n⚠ No port mappings found in change logs.")
        return True
    
    # Step 3: Load config to restore config file later
    config_file = project_root / config_path
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return False
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Display restoration mappings
    print(f"\n🔍 Port restoration mappings (from change logs):")
    for default_port in sorted(port_restore_map.keys()):
        modified_ports = sorted(port_restore_map[default_port])
        if modified_ports:
            ports_str = ', '.join(str(p) for p in modified_ports)
            print(f"   {default_port} ← [{ports_str}]")
    
    # Step 4: Process only files recorded in change logs
    print(f"\n📝 Restoring files (from change logs)...")
    
    total_files_restored = 0
    total_replacements = 0
    errors = []
    
    # Only process files that were recorded in change logs
    all_files = set(files_to_restore.keys())
    
    for file_rel_path in sorted(all_files):
        file_path = project_root / file_rel_path
        
        if not file_path.exists():
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            errors.append(f"Error reading {file_rel_path}: {e}")
            continue
        
        original_content = content
        file_replacement_count = 0
        
        # Only restore ports that were recorded for this specific file
        # Use regex to match standalone port numbers (not part of larger numbers)
        port_pairs = files_to_restore.get(file_rel_path, set())
        for default_port, modified_port in port_pairs:
            if modified_port == default_port:
                continue
                
            default_port_str = str(default_port)
            modified_port_str = str(modified_port)
            # Use regex: (?<![0-9]) = not preceded by digit, (?![0-9]) = not followed by digit
            pattern = rf'(?<![0-9]){re.escape(modified_port_str)}(?![0-9])'
            matches = re.findall(pattern, content)
            count = len(matches)
            
            if count > 0:
                content = re.sub(pattern, default_port_str, content)
                file_replacement_count += count
        
        if content != original_content:
            if not dry_run:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception as e:
                    errors.append(f"Error writing {file_rel_path}: {e}")
                    continue
            
            total_files_restored += 1
            total_replacements += file_replacement_count
            status = "[DRY RUN] Would restore" if dry_run else "Restored"
            print(f"   ✓ {status}: {file_rel_path} ({file_replacement_count} replacements)")
    
    # Step 5: Delete all change log files
    if not dry_run and changes_files:
        print(f"\n🗑️ Removing change logs...")
        for changes_file in changes_files:
            try:
                changes_file.unlink()
                print(f"   ✓ Removed: {changes_file.name}")
            except Exception as e:
                print(f"   ⚠ Could not remove {changes_file.name}: {e}")
    
    # Summary
    print()
    print("=" * 70)
    print(f"{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"   Files restored: {total_files_restored}")
    print(f"   Total replacements: {total_replacements}")
    
    if errors:
        print(f"\n⚠ Errors encountered ({len(errors)}):")
        for error in errors:
            print(f"   ✗ {error}")
        return False
    
    if total_files_restored > 0 or changes_files:
        print("\n✓ All ports restored to default values")
    else:
        print("\n✓ All ports are already at default values")
    
    return True


def auto_detect_changes_log(project_root: Path) -> Optional[Path]:
    """Auto-detect the changes log file to use for restoration."""
    changes_files = find_all_changes_logs(project_root)
    
    if not changes_files:
        return None
    
    if len(changes_files) == 1:
        return changes_files[0]
    
    # Multiple changes logs exist
    print("Multiple port changes detected. Please specify which to restore:")
    print()
    for i, f in enumerate(changes_files, 1):
        print(f"  {i}. {f.name}")
    print()
    print("Options:")
    print("  - Restore all: python global_preparation/apply_port_numbers.py --restore")
    print("  - List details: python global_preparation/apply_port_numbers.py --list")
    
    return None
