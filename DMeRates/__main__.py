"""python -m DMeRates — command-line utilities for DMeRates."""

from __future__ import annotations

import argparse
import sys


def _cmd_fetch_data(args: argparse.Namespace) -> None:
    from pathlib import Path
    from DMeRates.fetch_data import fetch_qcdark2

    dest = Path(args.dest) if args.dest else None
    print("Downloading QCDark2 dielectric files...")
    downloaded = fetch_qcdark2(
        materials=args.material or None,
        variants=args.variant or None,
        dest=dest,
        force=args.force,
    )
    noun = "file" if len(downloaded) == 1 else "files"
    print(f"\nDone. {len(downloaded)} {noun} downloaded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m DMeRates",
        description="DMeRates command-line utilities.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    fetch = sub.add_parser(
        "fetch-data",
        help="Download QCDark2 dielectric HDF5 files from GitHub.",
    )
    fetch.add_argument(
        "--material", "-m",
        nargs="+",
        metavar="MATERIAL",
        help=(
            "Materials to download. "
            "Choices: Si Ge GaAs SiC Diamond (case-insensitive). "
            "Default: all."
        ),
    )
    fetch.add_argument(
        "--variant", "-v",
        nargs="+",
        metavar="VARIANT",
        help=(
            "Variants to download. "
            "Choices: composite lfe nolfe. "
            "Default: all."
        ),
    )
    fetch.add_argument(
        "--dest",
        metavar="DIR",
        help=(
            "Root destination directory (must contain / will create "
            "dielectric_functions/). "
            "Default: form_factors/QCDark2/ inside this repo."
        ),
    )
    fetch.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download files that already exist.",
    )

    args = parser.parse_args()

    if args.command == "fetch-data":
        _cmd_fetch_data(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
