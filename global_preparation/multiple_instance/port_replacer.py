"""
PortReplacer class for applying and restoring port changes.
"""

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

from .constants import DEFAULT_PORTS
from .utils import calculate_port_mappings, calculate_port_offset, get_changes_log_name


class PortReplacer:
    """
    Handles port replacement operations for multi-instance deployments.
    
    This class provides methods to:
    - Apply port changes based on instance configuration
    - Detect current port values in files
    - Restore ports to default values
    """
    
    def __init__(self, config_path: str, dry_run: bool = False, 
                 instance_id: Optional[int] = None, instance_name: Optional[str] = None):
        self.config_path = config_path
        self.dry_run = dry_run
        self.instance_id = instance_id
        self.instance_name = instance_name
        self.changes_log = []
        self.project_root = Path(__file__).parent.parent.parent
        
        # Generate changes log path based on instance name
        if instance_name:
            changes_log_name = get_changes_log_name(instance_name)
        else:
            changes_log_name = "port_changes.json"
        self.changes_log_path = self.project_root / changes_log_name

    def load_config(self) -> Dict:
        """Load port configuration from YAML file."""
        config_file = self.project_root / self.config_path
        if not config_file.exists():
            print(f"Error: Config file not found: {config_file}")
            sys.exit(1)

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def save_config(self, config: Dict) -> None:
        """Save port configuration to YAML file."""
        config_file = self.project_root / self.config_path
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def validate_port_mappings(self, port_mappings: Dict[int, int]) -> bool:
        """Validate port mappings to ensure no conflicts."""
        new_ports = list(port_mappings.values())
        new_ports_set = set(new_ports)

        errors = []

        # Check for duplicate new ports
        if len(new_ports) != len(new_ports_set):
            duplicates = [p for p in new_ports_set if new_ports.count(p) > 1]
            errors.append(f"ERROR: Duplicate new ports found: {sorted(duplicates)}")

        if errors:
            for error in errors:
                print(error)
            return False

        return True

    def replace_in_file(self, file_path: Path, old_port: int, new_port: int) -> Tuple[bool, int]:
        """
        Replace old port with new port in a file.
        
        Uses regex to ensure we only replace standalone port numbers,
        not ports that are part of larger numbers (e.g., 2143 in 12143).
        """
        if not file_path.exists():
            return False, 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False, 0

        old_port_str = str(old_port)
        new_port_str = str(new_port)

        # Use regex to match standalone port numbers (not part of larger numbers)
        # (?<![0-9]) = not preceded by a digit
        # (?![0-9]) = not followed by a digit
        pattern = rf'(?<![0-9]){re.escape(old_port_str)}(?![0-9])'
        
        # Count matches
        matches = re.findall(pattern, content)
        count = len(matches)
        
        if count == 0:
            return False, 0

        new_content = re.sub(pattern, new_port_str, content)

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

    def detect_current_port_in_file(self, file_path: Path, default_port: int) -> Optional[int]:
        """
        Detect the current port value in a file.
        
        The port might be:
        - The default port (10001)
        - A modified port from another instance (11001, 12001, ...)
        - A manually set port
        
        Uses regex to ensure we only match standalone port numbers.
        
        Returns the detected port or None if not found.
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return None

        def port_exists(port: int) -> bool:
            """Check if port exists as a standalone number in content."""
            pattern = rf'(?<![0-9]){port}(?![0-9])'
            return bool(re.search(pattern, content))

        # First check if default port exists
        if port_exists(default_port):
            return default_port

        # Check for known offsets (instance 1-10)
        for offset in range(1000, 11000, 1000):
            modified_port = default_port + offset
            if port_exists(modified_port):
                return modified_port

        # Check for common manual offsets (100, 500, etc.)
        for offset in [100, 200, 500, 1500, 2000, 5000]:
            modified_port = default_port + offset
            if port_exists(modified_port):
                return modified_port

        return None

    def detect_current_port_across_files(self, files: List[str], default_port: int) -> Optional[int]:
        """
        Detect the current port value across multiple files.
        Returns the most common detected port.
        """
        detected_ports = []
        for file_rel_path in files:
            file_path = self.project_root / file_rel_path
            port = self.detect_current_port_in_file(file_path, default_port)
            if port is not None:
                detected_ports.append(port)
        
        if not detected_ports:
            return None
        
        # Return the most common port
        counter = Counter(detected_ports)
        return counter.most_common(1)[0][0]

    def apply_changes(self) -> bool:
        """Apply port number changes according to config."""
        config = self.load_config()
        files_by_port = config.get('files_by_port', {})

        # Calculate port mappings based on instance ID
        if self.instance_id:
            port_mappings = calculate_port_mappings(self.instance_id, DEFAULT_PORTS)
            instance_prefix = f"{self.instance_name}-"
            instance_suffix = f"-inst-{self.instance_name}"
            
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Applying port changes for instance: {self.instance_name} (ID: {self.instance_id})")
            print(f"Port offset: +{calculate_port_offset(self.instance_id)}")
        else:
            port_mappings = config.get('port_mappings', {})
            instance_prefix = config.get('instance_prefix', '')
            instance_suffix = config.get('instance_suffix', '')
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Applying port changes from config...")

        # Validate port mappings
        if not self.validate_port_mappings(port_mappings):
            sys.exit(1)

        print(f"Project root: {self.project_root}")
        print(f"Instance prefix: '{instance_prefix}'")
        print(f"Instance suffix: '{instance_suffix}'")
        print()

        total_replacements = 0
        unique_files_changed = set()  # Track unique files changed

        for default_port, port_info in files_by_port.items():
            default_port = int(default_port)
            target_port = port_mappings.get(default_port)

            if target_port is None:
                continue

            if default_port == target_port:
                continue

            comment = port_info.get('comment', f'Port {default_port}')
            files = port_info.get('files', [])

            # Detect current port in files (might be default or already modified)
            current_port = self.detect_current_port_across_files(files, default_port)
            
            if current_port is None:
                print(f"Port {default_port} → {target_port} ({comment})")
                print(f"  ⚠ Could not detect current port in files, using default")
                current_port = default_port
            elif current_port == target_port:
                print(f"Port {default_port} → {target_port} ({comment})")
                print(f"  ⚠ Already at target port, skipping")
                continue
            elif current_port != default_port:
                print(f"Port {default_port} → {target_port} ({comment})")
                print(f"  ℹ Current port detected: {current_port} (will be replaced)")
            else:
                print(f"Port {default_port} → {target_port} ({comment})")

            port_files_changed = 0
            port_replacements = 0

            for file_rel_path in files:
                file_path = self.project_root / file_rel_path
                # Replace current_port with target_port
                changed, count = self.replace_in_file(file_path, current_port, target_port)

                if changed:
                    port_files_changed += 1
                    port_replacements += count
                    unique_files_changed.add(file_rel_path)  # Track unique file
                    status = "[DRY RUN] Would change" if self.dry_run else "Changed"
                    print(f"  ✓ {status}: {file_rel_path} ({count} occurrences)")

                    # Record: default_port -> target_port (for restore)
                    self.changes_log.append({
                        'file': file_rel_path,
                        'default_port': default_port,
                        'current_port': current_port,
                        'new_port': target_port,
                        'count': count
                    })

            total_replacements += port_replacements
            print(f"  Summary: {port_files_changed} files, {port_replacements} replacements")
            print()

        print("=" * 70)
        print(f"Total: {len(unique_files_changed)} unique files changed, {total_replacements} replacements")

        if not self.dry_run and self.changes_log:
            # Save changes log with metadata
            with open(self.changes_log_path, 'w') as f:
                json.dump({
                    'instance_id': self.instance_id,
                    'instance_name': self.instance_name,
                    'instance_prefix': instance_prefix if self.instance_id else config.get('instance_prefix', ''),
                    'instance_suffix': instance_suffix if self.instance_id else config.get('instance_suffix', ''),
                    'port_offset': calculate_port_offset(self.instance_id) if self.instance_id else None,
                    'applied_at': datetime.now().isoformat(),
                    'changes': self.changes_log
                }, f, indent=2)
            print(f"\nChanges logged to: {self.changes_log_path}")
            print("\nTo restore original ports, run:")
            print(f"  python global_preparation/apply_port_numbers.py --restore")

        return len(unique_files_changed) > 0

    def restore_changes(self, use_git: bool = False) -> bool:
        """Restore original port numbers."""
        print("Restoring original port numbers...")
        print()

        if not self.changes_log_path.exists():
            print(f"Error: Changes log not found: {self.changes_log_path}")
            print("\nAvailable options:")
            print(f"  - List available: python global_preparation/apply_port_numbers.py --list")
            print(f"  - Restore all:    python global_preparation/apply_port_numbers.py --restore")
            print(f"  - Use git:        python global_preparation/apply_port_numbers.py --restore --use-git")
            return False

        with open(self.changes_log_path, 'r') as f:
            log_data = json.load(f)

        print(f"Found changes log: {self.changes_log_path}")
        print(f"Instance: {log_data.get('instance_name', 'unknown')} (ID: {log_data.get('instance_id', 'unknown')})")
        print(f"Applied at: {log_data.get('applied_at', 'unknown')}")
        print(f"Number of changes: {len(log_data.get('changes', []))}")
        print()

        if use_git:
            return self._restore_with_git()
        else:
            return self._restore_with_reverse_replacement(log_data)

    def _restore_with_reverse_replacement(self, log_data: Dict) -> bool:
        """Restore original port numbers by performing reverse replacement."""
        print("Using reverse replacement method...")
        print()

        changes = log_data.get('changes', [])
        if not changes:
            print("No changes recorded in log file")
            return True

        # Group changes by file
        files_to_restore: Dict[str, List[Dict]] = {}
        for change in changes:
            file_path = change['file']
            if file_path not in files_to_restore:
                files_to_restore[file_path] = []
            files_to_restore[file_path].append(change)

        total_files_restored = 0
        total_replacements = 0
        errors = []

        for file_rel_path, file_changes in files_to_restore.items():
            file_path = self.project_root / file_rel_path
            
            if not file_path.exists():
                print(f"  ⚠ File not found (skipped): {file_rel_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                errors.append(f"Error reading {file_rel_path}: {e}")
                continue

            original_content = content
            file_replacement_count = 0

            for change in file_changes:
                default_port = change.get('default_port', change.get('old_port'))
                new_port = change['new_port']
                
                new_port_str = str(new_port)
                default_port_str = str(default_port)
                
                # Use regex to match standalone port numbers (not part of larger numbers)
                pattern = rf'(?<![0-9]){re.escape(new_port_str)}(?![0-9])'
                matches = re.findall(pattern, content)
                count = len(matches)
                
                if count > 0:
                    content = re.sub(pattern, default_port_str, content)
                    file_replacement_count += count

            if content != original_content:
                if not self.dry_run:
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    except Exception as e:
                        errors.append(f"Error writing {file_rel_path}: {e}")
                        continue

                total_files_restored += 1
                total_replacements += file_replacement_count
                status = "[DRY RUN] Would restore" if self.dry_run else "Restored"
                print(f"  ✓ {status}: {file_rel_path} ({file_replacement_count} replacements)")

        print()
        print("=" * 70)
        print(f"{'[DRY RUN] ' if self.dry_run else ''}Total: {total_files_restored} files restored, {total_replacements} replacements")

        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(f"  ✗ {error}")

        if not self.dry_run and total_files_restored > 0:
            try:
                self.changes_log_path.unlink()
                print(f"\n✓ Removed changes log: {self.changes_log_path}")
            except Exception as e:
                print(f"\n⚠ Warning: Could not remove changes log: {e}")

            print("\n✓ Port numbers restored to original values")

        return len(errors) == 0

    def _restore_with_git(self) -> bool:
        """Restore using git restore."""
        print("Using git restore method...")
        print("⚠ WARNING: This will restore ALL modified files!")
        print()

        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("Error: Not in a git repository")
                return False

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

            if self.dry_run:
                print("[DRY RUN] Would restore the following files:")
                for line in modified_files:
                    print(f"  {line}")
                return True

            result = subprocess.run(
                ['git', 'restore', '.'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✓ Successfully restored files")

                if self.changes_log_path.exists():
                    self.changes_log_path.unlink()
                    print(f"✓ Removed changes log: {self.changes_log_path}")

                return True
            else:
                print(f"Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error: {e}")
            return False
