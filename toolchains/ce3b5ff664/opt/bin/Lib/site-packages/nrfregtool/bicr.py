#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

import dataclasses as dc
from collections import defaultdict
from functools import cached_property
from pprint import pformat
from typing import List, NamedTuple, Optional, Union

from devicetree import edtlib

from .common import Record, log_dbg, log_dev, log_vrb
from .parsed_dt import (
    ExtractedConfig,
    InterpretedPeripheral,
    ParsedDTNode,
    PeripheralResult,
    PropertyMap,
    sort_configs_by_type,
)

BICR_COMPATIBLE = "nordic,nrf-bicr"


# Registers that must be set as unconstrained
BICR_UNCONSTRAINED = (
    "lfosc_lfxoconfig_0",
    "hfxo_startuptime_0",
    "hfxo_config_0",
)


class SvdEnum:
    """Helper class for constructing commonly used SVD enums."""

    UNCONFIGURED = "Unconfigured"

    @classmethod
    def int_or_unconfigured(cls, val: Optional[int]) -> Union[int, str]:
        return val if isinstance(val, int) else cls.UNCONFIGURED

    INTERNAL = "Internal"
    EXTERNAL = "External"
    SHORTED = "Shorted"

    @classmethod
    def rail_connection(cls, rail: str) -> str:
        for connection in [cls.INTERNAL, cls.EXTERNAL, cls.SHORTED]:
            if rail.lower() == connection.lower():
                return connection
        return cls.UNCONFIGURED

    ENABLED = "Enabled"
    DISABLED = "Disabled"

    @classmethod
    def enabled_if(cls, is_enabled: bool) -> str:
        return cls.ENABLED if is_enabled else cls.DISABLED

    NOT_PRESENT = "NotPresent"
    PRESENT = "Present"

    @classmethod
    def present_if(cls, is_present: bool) -> str:
        return cls.PRESENT if is_present else cls.NOT_PRESENT

    LFXO = "LFXO"
    LFRC = "LFRC"
    SYNTH = "Synth"

    @classmethod
    def lfosc_source(cls, source: str) -> str:
        for clksrc in [cls.LFXO, cls.LFRC, cls.SYNTH]:
            if source.lower() == clksrc.lower():
                return clksrc
        return cls.UNCONFIGURED

    CRYSTAL = "Crystal"
    PIERCE = "Pierce"
    EXTSQUARE = "ExtSquare"
    EXTSINE = "ExtSine"
    AUTO = "Auto"

    @classmethod
    def clk_mode(cls, mode: str) -> str:
        for binding, clkmode in [
            ("crystal", cls.CRYSTAL),
            ("external-square", cls.EXTSQUARE),
            ("external-sine", cls.EXTSINE),
            ("auto", cls.AUTO),
        ]:
            if mode.lower() == binding:
                return clkmode
        return cls.UNCONFIGURED

    @classmethod
    def lfosc_accuracy(cls, accuracy: Optional[int]) -> str:
        supported_ppms = [20, 30, 50, 75, 100, 150, 250, 500]

        if accuracy is None:
            return cls.UNCONFIGURED

        if accuracy not in supported_ppms:
            raise ValueError(
                f"LF oscillator accuracy must be one of {supported_ppms}; got {accuracy}"
            )

        return f"{accuracy}ppm"

    @classmethod
    def capacitance(cls, value: Optional[int]) -> Union[int, str]:
        if value is None:
            return cls.UNCONFIGURED

        return cls.EXTERNAL if value == 0 else value

    DISCONNECTED = "Disconnected"
    EXTERNAL_1V8 = "External1V8"
    EXTERNAL_3V = "External3V"
    EXTERNAL_FULL = "ExternalFull"

    @classmethod
    def port_rail(cls, value: int):
        connections = [
            cls.DISCONNECTED,
            cls.SHORTED,
            cls.EXTERNAL_1V8,
            cls.EXTERNAL_3V,
            cls.EXTERNAL_FULL,
        ]
        try:
            return connections[value]
        except IndexError:
            return cls.UNCONFIGURED

    OHMS = "Ohms"

    @classmethod
    def port_drive(cls, ohms: int):
        return f"{cls.OHMS}{ohms}"


@dc.dataclass(frozen=True, order=True)
class PowerConfig(ExtractedConfig):
    """Power rail configuration."""

    vddao0v8: str
    vdd1v0: str
    vddrf1v0: str
    vddao1v8: str
    vddao5v0: str
    vddvs0v8: str
    inductor: bool

    def to_record(self) -> Record:
        record = Record(
            {
                "power_config_0": {
                    "VDDAO0V8": SvdEnum.rail_connection(self.vddao0v8),
                    "VDD1V0": SvdEnum.rail_connection(self.vdd1v0),
                    "VDDRF1V0": SvdEnum.rail_connection(self.vddrf1v0),
                    "VDDAO1V8": SvdEnum.rail_connection(self.vddao1v8),
                    "VDDAO5V0": SvdEnum.rail_connection(self.vddao5v0),
                    "VDDVS0V8": SvdEnum.rail_connection(self.vddvs0v8),
                    "INDUCTOR": SvdEnum.present_if(self.inductor),
                }
            }
        )

        return record


class IOPortRail(NamedTuple):
    """Helper tuple for IOPORT rail configurations."""

    port: int
    rail: int


class IOPortDrive(NamedTuple):
    """Helper tuple for IOPORT drive configurations."""

    port: int
    resistance: int


@dc.dataclass(frozen=True, order=True)
class IOPortConfig(ExtractedConfig):
    """GPIO port configuration."""

    rails: List[IOPortRail] = dc.field(default_factory=list)
    drives: List[IOPortDrive] = dc.field(default_factory=list)

    def to_record(self) -> Record:
        records = defaultdict(dict)

        reg_idx = lambda c: "1" if c.port > 7 else "0"

        for cfg in self.rails:
            records[f"ioport_power{reg_idx(cfg)}_0"].update(
                {
                    f"P{cfg.port}": SvdEnum.port_rail(cfg.rail),
                }
            )

        for cfg in self.drives:
            records[f"ioport_drivectrl{reg_idx(cfg)}_0"].update(
                {
                    f"P{cfg.port}": SvdEnum.port_drive(cfg.resistance),
                }
            )

        return Record(records)


class LFAutoCal(NamedTuple):
    """Helper tuple for low-frequency oscillator autocalibration configurations."""

    temp_interval: int
    temp_delta: int
    max_num_intervals: int
    enabled: bool


@dc.dataclass(frozen=True, order=True)
class LFConfig(ExtractedConfig):
    """Low-frequency oscillator configuration."""

    src: str
    mode: str
    accuracy: Optional[int] = None
    loadcap: Optional[int] = None
    startup: Optional[int] = None
    autocalibration: Optional[LFAutoCal] = None

    def to_record(self) -> Record:
        record = Record()

        if self.src:
            record.update(
                {
                    "lfosc_config_0": {"SRC": SvdEnum.lfosc_source(self.src)},
                }
            )

        if any(
            x and x is not None
            for x in [self.accuracy, self.mode, self.loadcap, self.startup]
        ):
            record.update(
                {
                    "lfosc_lfxoconfig_0": {
                        "ACCURACY": SvdEnum.lfosc_accuracy(self.accuracy),
                        "MODE": SvdEnum.clk_mode(self.mode),
                        "LOADCAP": SvdEnum.capacitance(self.loadcap),
                        "TIME": SvdEnum.int_or_unconfigured(self.startup),
                    }
                }
            )

        if self.autocalibration:
            record.update(
                {
                    "lfosc_lfrcautocalconfig_0": {
                        "TEMPINTERVAL": self.autocalibration.temp_interval,
                        "TEMPDELTA": self.autocalibration.temp_delta,
                        "INTERVALMAXNO": self.autocalibration.max_num_intervals,
                        "ENABLE": SvdEnum.enabled_if(self.autocalibration.enabled),
                    }
                }
            )

        return record


@dc.dataclass(frozen=True, order=True)
class HFConfig(ExtractedConfig):
    """High-frequency oscillator configuration."""

    mode: str
    loadcap: Optional[int] = None
    startup: Optional[int] = None

    def to_record(self) -> Record:
        record = Record()

        if any(x and x is not None for x in [self.mode, self.loadcap]):
            record.update(
                {
                    "hfxo_config_0": {
                        "MODE": SvdEnum.clk_mode(self.mode),
                        "LOADCAP": SvdEnum.capacitance(self.loadcap),
                    }
                }
            )

        if self.startup:
            record.update(
                {
                    "hfxo_startuptime_0": {
                        "TIME": SvdEnum.int_or_unconfigured(self.startup),
                    }
                }
            )

        return record


@dc.dataclass(frozen=True, order=True)
class TamperConfig(ExtractedConfig):
    """Tamper controller configuration."""

    switch_enabled: bool
    shield_channels: List[int] = dc.field(default_factory=list)

    def to_record(self) -> Record:
        record = Record()

        if self.switch_enabled:
            record.update(
                {
                    "tampc_tamperswitch_0": {
                        "TAMPERSWITCH": SvdEnum.ENABLED,
                    }
                }
            )

        if self.shield_channels:
            record.update(
                {
                    "tampc_activeshield_0": {
                        f"CHEN_{ch}": SvdEnum.ENABLED for ch in self.shield_channels
                    }
                }
            )

        return record


class InterpretedBICR(InterpretedPeripheral):
    """
    BICR-related information interpreted from the parsed devicetree nodes that are relevant to the
    peripheral's configurations.
    """

    def __init__(self, devicetree: edtlib.EDT):
        """
        Initialize the class attribute(s).

        :param devicetree: edtlib Devicetree object
        """
        super().__init__(devicetree=devicetree, name="BICR")

        nodes = (ParsedDTNode(n) for n in devicetree.nodes)
        matching_nodes = [n for n in nodes if n.compatibles_contain(BICR_COMPATIBLE)]
        if not matching_nodes:
            raise RuntimeError(f"No BICR node with compatible {BICR_COMPATIBLE} found.")
        if len(matching_nodes) != 1:
            raise RuntimeError(
                f"Expected exactly one BICR node with compatible {BICR_COMPATIBLE}. "
                f"Found {len(matching_nodes)} nodes: {matching_nodes}."
            )

        self._node: ParsedDTNode = matching_nodes[0]
        self._properties.update(self._node.properties)

    def __str__(self) -> str:
        """String representation of the class."""

        props = {
            "Properties": [str(v) for v in self.properties.values()],
            "Chosen": self.chosen,
        }
        return f"Parsed BICR:\n{pformat(props)}\n"

    @property
    def address(self) -> int:
        """
        Base address of the BICR as defined by the devicetree.
        """
        address = self._node.address
        if address is None:
            raise RuntimeError(f"Missing 'reg' property in BICR node {self._node}")
        return address

    @cached_property
    def power_props(self) -> PropertyMap:
        """
        Mapping of properties related to BICR power configurations.
        """
        properties = PropertyMap(
            {
                k: v
                for k, v in self.properties.items()
                if any(k.startswith(x) for x in ["power", "inductor"])
            }
        )
        return properties

    @cached_property
    def lfosc_props(self) -> PropertyMap:
        """
        Mapping of properties related to low-frequency oscillator configurations.
        """
        properties = PropertyMap(
            {k: v for k, v in self.properties.items() if k.startswith("lf")}
        )
        return properties

    @cached_property
    def hfxo_props(self) -> PropertyMap:
        """
        Mapping of properties related to High-Frequency Clock configurations.
        """
        properties = PropertyMap(
            {k: v for k, v in self.properties.items() if k.startswith("hfxo")}
        )
        return properties

    @cached_property
    def tampc_props(self) -> PropertyMap:
        """
        Mapping of properties related to Tamper Controller configurations.
        """
        properties = PropertyMap(
            {k: v for k, v in self.properties.items() if k.startswith("tamper")}
        )
        return properties

    @cached_property
    def ioport_props(self) -> PropertyMap:
        """
        Mapping of properties related to GPIO port configurations.
        """
        properties = PropertyMap(
            {k: v for k, v in self.properties.items() if k.startswith("ioport")}
        )
        return properties


def from_dts(devicetree: edtlib.EDT) -> PeripheralResult:
    """
    Gather information from parsed devicetree nodes on configurations that must set in BICR
    in order to be configured on boot, then map those records into a memory map of BICR
    contents.

    :param devicetree: Devicetree structure.

    :return: Meta-record of extracted BICR records from devicetree nodes.
    """

    configs: List[ExtractedConfig] = []
    records: List[Record] = []

    bicr = InterpretedBICR(devicetree)

    log_dev(f"Parsed DTS nodes:\n{str(bicr)}")
    log_dev("")

    configs.append(extract_power_configs(bicr.power_props))
    configs.append(extract_ioport_configs(bicr.ioport_props))
    configs.append(extract_lfosc_configs(bicr.lfosc_props))
    configs.append(extract_hfxo_configs(bicr.hfxo_props))
    configs.append(extract_tamper_configs(bicr.tampc_props))

    for config_list in sort_configs_by_type(configs):
        for config in config_list:
            records.append(config.to_record())

    log_dev(f"All DTS records for BICR (pre-filter):")
    log_dev(pformat(records))
    log_dev("")

    bicr_record = Record()
    for record in [x for x in records if x]:
        bicr_record.update(record)

    log_dbg(f"Interpreted DTS records for BICR:")
    log_vrb(pformat(bicr_record))
    log_vrb("")

    return PeripheralResult(base_address=bicr.address, record=bicr_record)


def extract_power_configs(
    properties: PropertyMap,
) -> PowerConfig:
    """
    Extract power-related BICR configurations.

    :param properties: Power properties kept interpreted from devicetree's BICR.

    :return: List of all extracted configurations.
    """

    config = PowerConfig(
        vdd1v0=properties.get_val("power-vdd1v0", ""),
        vddao0v8=properties.get_val("power-vddao0v8", ""),
        vddao1v8=properties.get_val("power-vddao1v8", ""),
        vddao5v0=properties.get_val("power-vddao5v0", ""),
        vddrf1v0=properties.get_val("power-vddrf1v0", ""),
        vddvs0v8=properties.get_val("power-vddvs0v8", ""),
        inductor=properties.get_val("inductor-present", False),
    )

    return config


def extract_ioport_configs(
    properties: PropertyMap,
) -> PowerConfig:
    """
    Extract GPIO port-related BICR configurations.

    :param properties: GPIO port properties kept interpreted from devicetree's BICR.

    :return: List of all extracted configurations.
    """

    config = IOPortConfig(
        rails=[
            IOPortRail(port=entry.controller.props["port"].val, rail=entry.data["rail"])
            for entry in properties.get_val("ioport-power-rails", [])
        ],
        drives=[
            IOPortDrive(
                port=entry.controller.props["port"].val,
                resistance=entry.data["resistance"],
            )
            for entry in properties.get_val("ioport-drivectrls", [])
        ],
    )

    return config


def extract_lfosc_configs(
    properties: PropertyMap,
) -> PowerConfig:
    """
    Extract low-frequency oscillator-related BICR configurations.

    :param properties: LFOSC properties kept interpreted from devicetree's BICR.

    :return: List of all extracted configurations.
    """

    autocal_array = properties.get_val("lfrc-autocalibration")

    config = LFConfig(
        src=properties.get_val("lfosc-src", ""),
        mode=properties.get_val("lfosc-mode", ""),
        accuracy=properties.get_val("lfosc-accuracy"),
        loadcap=properties.get_val("lfosc-loadcap"),
        startup=properties.get_val("lfosc-startup"),
        autocalibration=(
            LFAutoCal(
                enabled=True,
                temp_interval=autocal_array[0],
                temp_delta=autocal_array[1],
                max_num_intervals=autocal_array[2],
            )
            if autocal_array
            else None
        ),
    )

    return config


def extract_hfxo_configs(
    properties: PropertyMap,
) -> PowerConfig:
    """
    Extract high-frequency oscillator-related BICR configurations.

    :param properties: HFXO properties kept interpreted from devicetree's BICR.

    :return: List of all extracted configurations.
    """

    config = HFConfig(
        mode=properties.get_val("hfxo-mode", ""),
        loadcap=properties.get_val("hfxo-loadcap"),
        startup=properties.get_val("hfxo-startup"),
    )

    return config


def extract_tamper_configs(
    properties: PropertyMap,
) -> PowerConfig:
    """
    Extract tamper controller (TAMPC)-related BICR configurations.

    :param properties: TAMPC properties kept interpreted from devicetree's BICR.

    :return: List of all extracted configurations.
    """

    config = TamperConfig(
        switch_enabled=properties.get_val("tamperswitch-enable", False),
        shield_channels=properties.get_val("tamper-activeshield-channels", []),
    )

    return config
