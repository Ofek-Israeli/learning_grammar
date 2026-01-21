#!/usr/bin/env python3
"""
Interactive Kconfig menuconfig for Bloat Axis GEPA.

Usage:
    python kmenuconfig.py                    # Interactive TUI
    python kmenuconfig.py --help             # Show help
    python kmenuconfig.py defconfig          # Load default config
    python kmenuconfig.py savedefconfig      # Save current as defconfig

Requires: kconfiglib (pip install kconfiglib)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure we're in the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

# Try to import kconfiglib
try:
    import kconfiglib
except ImportError:
    print("Error: kconfiglib not installed.")
    print("Install with: pip install kconfiglib")
    sys.exit(1)


def load_kconfig() -> kconfiglib.Kconfig:
    """Load the Kconfig file."""
    kconfig_path = SCRIPT_DIR / "Kconfig"
    if not kconfig_path.exists():
        print(f"Error: Kconfig not found at {kconfig_path}")
        sys.exit(1)
    
    return kconfiglib.Kconfig(str(kconfig_path))


def cmd_menuconfig():
    """Run interactive menuconfig (TUI)."""
    try:
        import menuconfig
    except ImportError:
        print("Error: menuconfig module not available.")
        print("Install with: pip install kconfiglib")
        print("Or use: python -m menuconfig Kconfig")
        sys.exit(1)
    
    kconf = load_kconfig()
    
    # Load existing .config if present
    config_path = SCRIPT_DIR / ".config"
    if config_path.exists():
        kconf.load_config(str(config_path))
    
    # Run menuconfig
    menuconfig.menuconfig(kconf)
    
    # Save config
    kconf.write_config(str(config_path))
    print(f"Configuration saved to {config_path}")


def cmd_defconfig():
    """Load default configuration."""
    kconf = load_kconfig()
    
    defconfig_path = SCRIPT_DIR / "defconfig"
    if defconfig_path.exists():
        kconf.load_config(str(defconfig_path))
        print(f"Loaded defaults from {defconfig_path}")
    else:
        print("No defconfig found, using Kconfig defaults")
    
    # Save as .config
    config_path = SCRIPT_DIR / ".config"
    kconf.write_config(str(config_path))
    print(f"Configuration saved to {config_path}")


def cmd_savedefconfig():
    """Save current .config as defconfig."""
    kconf = load_kconfig()
    
    config_path = SCRIPT_DIR / ".config"
    if config_path.exists():
        kconf.load_config(str(config_path))
    
    defconfig_path = SCRIPT_DIR / "defconfig"
    kconf.write_min_config(str(defconfig_path))
    print(f"Minimal config saved to {defconfig_path}")


def cmd_oldconfig():
    """Update .config with new options from Kconfig (non-interactive)."""
    kconf = load_kconfig()
    
    config_path = SCRIPT_DIR / ".config"
    if config_path.exists():
        kconf.load_config(str(config_path))
    
    # Write updated config
    kconf.write_config(str(config_path))
    print(f"Configuration updated at {config_path}")


def cmd_alldefconfig():
    """Set all options to default values."""
    kconf = load_kconfig()
    
    # Use defaults
    for sym in kconf.unique_defined_syms:
        sym.set_value(sym.str_value if sym.str_value else 2)
    
    config_path = SCRIPT_DIR / ".config"
    kconf.write_config(str(config_path))
    print(f"All defaults saved to {config_path}")


def cmd_show():
    """Show current configuration."""
    kconf = load_kconfig()
    
    config_path = SCRIPT_DIR / ".config"
    if config_path.exists():
        kconf.load_config(str(config_path))
    
    print("Current configuration:")
    print("-" * 40)
    
    for sym in kconf.unique_defined_syms:
        if sym.user_value is not None:
            name = sym.name
            value = sym.str_value
            print(f"  {name} = {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Kconfig menuconfig for Bloat Axis GEPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)          Run interactive menuconfig TUI
  defconfig       Load default configuration
  savedefconfig   Save current config as minimal defconfig
  oldconfig       Update .config with new Kconfig options
  alldefconfig    Reset all to Kconfig defaults
  show            Display current configuration
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["defconfig", "savedefconfig", "oldconfig", "alldefconfig", "show"],
        help="Configuration command"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        # Default: run menuconfig
        cmd_menuconfig()
    elif args.command == "defconfig":
        cmd_defconfig()
    elif args.command == "savedefconfig":
        cmd_savedefconfig()
    elif args.command == "oldconfig":
        cmd_oldconfig()
    elif args.command == "alldefconfig":
        cmd_alldefconfig()
    elif args.command == "show":
        cmd_show()


if __name__ == "__main__":
    main()
