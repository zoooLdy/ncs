#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

if not __package__:
    raise RuntimeError(
        f"Entrypoint {__file__} is only usable when installed as a package."
    )


import importlib.metadata
from .cli import cli
from .__init__ import PACKAGE_NAME
from more_itertools import first


def package_main():
    """
    Main entry point for the tool when installed as a package.
    """
    executable_name = first(
        importlib.metadata.distribution(PACKAGE_NAME).entry_points
    ).name
    cli(prog_name=executable_name)


if __name__ == "__main__":
    package_main()
