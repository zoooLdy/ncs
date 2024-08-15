#!/bin/python3
#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

import sys
import pathlib


def script_main():
    """
    Main entry point of the tool when called as a standalone script.

    This modifies the sys.path to include the parent directory as a package, which allows
    relative imports to be accepted when executed this way.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1].absolute()))

    from nrfregtool.cli import cli

    # Here we hardcode the expected executable name, as there doesn't appear to be a great way of
    # extracting this automatically when run as a script, regardless of the sys.path modification.
    # However, it allows version prints to align with one another regardless of the tool being
    # installed as a package or not.
    cli(prog_name="nrf-regtool")


if __name__ == "__main__":
    script_main()
