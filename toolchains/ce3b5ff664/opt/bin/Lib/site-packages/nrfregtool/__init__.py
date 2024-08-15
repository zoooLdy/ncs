#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

import importlib.metadata


PACKAGE_NAME = "nrf-regtool"


try:
    __version__ = importlib.metadata.version(PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    # Package is not installed, add package directory to path to get version info and allow
    # relative imports to follow when run as a script.
    import setuptools_scm
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).parents[1].absolute()))

    __version__ = setuptools_scm.get_version(root="..", relative_to=__file__)


from .common import (
    AddressRegion,
    AddressOffset,
    DomainID,
    OwnerID,
    ProcessorID,
    Record,
    config_log,
    log_vrb,
    log_dbg,
    log_err,
)
from .core import (
    LogicalPeripheral,
    parse_toml,
    parse_dts,
    field_as_toml,
    register_as_toml,
    generate_toml,
    generate_hex,
    split_register_path,
    split_hex_diffs,
)
from .parsed_dt import (
    ParsedDTNode,
    ProcessorInfo,
    dt_processor_info,
)
from .uicr import (
    GpioPin,
    IpcMapConfig,
    SourceMapIndex,
    SinkMapIndex,
    NodeType,
    ResourceCompatible,
    InterpretedUICR,
    extract_memory_configs,
    extract_periph_configs,
    extract_gpio_configs,
    extract_gpiote_configs,
    extract_grtc_configs,
    extract_dppic_configs,
    extract_ipct_configs,
    extract_vtor_configs,
    extract_ptrextuicr_config,
)
from .bicr import (
    InterpretedBICR,
    PowerConfig,
    extract_power_configs,
)
