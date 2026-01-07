#!/usr/bin/env python3
"""
Apply port number changes across the Toolathlon codebase.

This script reads port mappings from configs/ports_config.yaml and replaces
port numbers in specified files to avoid conflicts when running multiple
Toolathlon instances on the same machine.

Usage:
    # Apply port changes
    python global_preparation/apply_port_numbers.py

    # Dry run (show what would change)
    python global_preparation/apply_port_numbers.py --dry-run

    # Restore original ports (requires clean git state)
    python global_preparation/apply_port_numbers.py --restore

    # Use custom config file
    python global_preparation/apply_port_numbers.py --config custom_ports.yaml
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


class PortReplacer:
    def __init__(self, config_path: str, dry_run: bool = False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.changes_log = []
        self.project_root = Path(__file__).parent.parent
        self.changes_log_path = self.project_root / "port_changes.json"

    def load_config(self) -> Dict:
        """Load port configuration from YAML file."""
        config_file = self.project_root / self.config_path
        if not config_file.exists():
            print(f"Error: Config file not found: {config_file}")
            sys.exit(1)

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def validate_port_mappings(self, port_mappings: Dict[int, int]) -> bool:
        """
        Validate port mappings to ensure no conflicts.

        Checks:
        1. No new port equals any old port
        2. No duplicate new ports

        Returns True if valid, exits with error if invalid.
        """
        old_ports = set(port_mappings.keys())
        new_ports = list(port_mappings.values())
        new_ports_set = set(new_ports)

        errors = []

        # Check 1: New port conflicts with any old port
        conflicts_with_old = new_ports_set & old_ports
        if conflicts_with_old:
            errors.append(f"ERROR: New ports conflict with original ports: {sorted(conflicts_with_old)}")
            errors.append("  This would cause incorrect replacements!")
            for port in sorted(conflicts_with_old):
                # Find which new port mapping causes this conflict
                for old, new in port_mappings.items():
                    if new == port:
                        errors.append(f"    - Port {old} → {new} conflicts with original port {port}")

        # Check 2: Duplicate new ports
        if len(new_ports) != len(new_ports_set):
            duplicates = [p for p in new_ports_set if new_ports.count(p) > 1]
            errors.append(f"ERROR: Duplicate new ports found: {sorted(duplicates)}")
            for dup_port in sorted(duplicates):
                old_ports_mapping_to_dup = [old for old, new in port_mappings.items() if new == dup_port]
                errors.append(f"    - Port {dup_port} is used by: {old_ports_mapping_to_dup}")

        if errors:
            print("=" * 70)
            print("PORT MAPPING VALIDATION FAILED")
            print("=" * 70)
            for error in errors:
                print(error)
            print()
            print("Please fix the port mappings in your config file:")
            print(f"  {self.project_root / self.config_path}")
            print("=" * 70)
            return False

        return True

    def replace_in_file(self, file_path: Path, old_port: int, new_port: int) -> Tuple[bool, int]:
        """
        Replace old port with new port in a file.
        Returns (changed, num_replacements).
        """
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return False, 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False, 0

        # Use word boundary to avoid partial matches
        # This will match the port number when it appears as a complete number
        old_port_str = str(old_port)
        new_port_str = str(new_port)

        # Count occurrences
        count = content.count(old_port_str)
        if count == 0:
            return False, 0

        # Replace all occurrences
        new_content = content.replace(old_port_str, new_port_str)

        if new_content == content:
            return False, 0

        if not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False, 0

        return True, count

    def apply_changes(self):
        """Apply port number changes according to config."""
        config = self.load_config()
        port_mappings = config.get('port_mappings', {})
        files_by_port = config.get('files_by_port', {})

        # Validate port mappings before applying
        if not self.validate_port_mappings(port_mappings):
            sys.exit(1)

        print(f"{'[DRY RUN] ' if self.dry_run else ''}Applying port changes...")
        print(f"Project root: {self.project_root}")
        print(f"Config: {self.config_path}")
        print()

        total_files_changed = 0
        total_replacements = 0

        for old_port, port_info in files_by_port.items():
            old_port = int(old_port)
            new_port = port_mappings.get(old_port)

            if new_port is None:
                print(f"Warning: No mapping for port {old_port}, skipping")
                continue

            if old_port == new_port:
                print(f"Port {old_port}: No change needed (same as target)")
                continue

            comment = port_info.get('comment', f'Port {old_port}')
            files = port_info.get('files', [])

            print(f"Port {old_port} → {new_port} ({comment})")

            files_changed = 0
            port_replacements = 0

            for file_rel_path in files:
                file_path = self.project_root / file_rel_path
                changed, count = self.replace_in_file(file_path, old_port, new_port)

                if changed:
                    files_changed += 1
                    port_replacements += count
                    status = "[DRY RUN] Would change" if self.dry_run else "Changed"
                    print(f"  ✓ {status}: {file_rel_path} ({count} occurrences)")

                    self.changes_log.append({
                        'file': file_rel_path,
                        'old_port': old_port,
                        'new_port': new_port,
                        'count': count
                    })

            total_files_changed += files_changed
            total_replacements += port_replacements
            print(f"  Summary: {files_changed} files, {port_replacements} replacements")
            print()

        print("=" * 70)
        print(f"Total: {total_files_changed} files changed, {total_replacements} replacements")

        if not self.dry_run and self.changes_log:
            # Save changes log
            with open(self.changes_log_path, 'w') as f:
                json.dump({
                    'config_used': self.config_path,
                    'changes': self.changes_log
                }, f, indent=2)
            print(f"\nChanges logged to: {self.changes_log_path}")
            print("\nTo restore original ports, run:")
            print("  git restore .")
            print("  # or")
            print(f"  python {__file__} --restore")

        return total_files_changed > 0

    def restore_changes(self):
        """Restore original port numbers using git."""
        print("Restoring original port numbers...")
        print()

        # Check if we have a changes log
        if self.changes_log_path.exists():
            with open(self.changes_log_path, 'r') as f:
                log_data = json.load(f)

            print(f"Found changes log: {self.changes_log_path}")
            print(f"Config used: {log_data.get('config_used')}")
            print(f"Number of changes: {len(log_data.get('changes', []))}")
            print()

        # Use git to restore
        import subprocess

        try:
            # Check if git is available and we're in a git repo
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("Error: Not in a git repository")
                print("Please manually restore files or use a backup")
                return False

            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            modified_files = [line for line in result.stdout.split('\n') if line.strip()]

            if not modified_files:
                print("No modified files to restore")
                return True

            print(f"Found {len(modified_files)} modified files")
            print("\nRestoring files with git...")

            # Restore all modified files
            result = subprocess.run(
                ['git', 'restore', '.'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✓ Successfully restored original port numbers")

                # Remove changes log
                if self.changes_log_path.exists():
                    self.changes_log_path.unlink()
                    print(f"✓ Removed changes log: {self.changes_log_path}")

                return True
            else:
                print(f"Error restoring files: {result.stderr}")
                return False

        except FileNotFoundError:
            print("Error: git command not found")
            print("Please install git or manually restore files")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Apply or restore port number changes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply changes (default)
  python global_preparation/apply_port_numbers.py

  # Dry run to preview changes
  python global_preparation/apply_port_numbers.py --dry-run

  # Restore original ports
  python global_preparation/apply_port_numbers.py --restore

  # Use custom config
  python global_preparation/apply_port_numbers.py --config my_ports.yaml
        """
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
        help='Restore original port numbers using git'
    )

    args = parser.parse_args()

    replacer = PortReplacer(args.config, dry_run=args.dry_run)

    if args.restore:
        success = replacer.restore_changes()
        sys.exit(0 if success else 1)
    else:
        success = replacer.apply_changes()
        sys.exit(0 if success or args.dry_run else 1)


if __name__ == '__main__':
    main()
