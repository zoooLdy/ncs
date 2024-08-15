#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import dataclasses as dc
import enum
import re
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, groupby
from pprint import pformat
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple, Union

from devicetree import edtlib

from .common import (
    AddressRegion,
    DomainID,
    OwnerID,
    ProcessorID,
    Record,
    else_none,
    gpio_port_pin_decode,
    log_dbg,
    log_dev,
    log_vrb,
    nrf_psel_decode,
    verbose_logs,
)
from .ctrlsel import CTRLSEL_DEFAULT, dt_lookup_ctrlsel
from .parsed_dt import (
    PARTITION_COMPAT_REGEX,
    ExtractedConfig,
    IndexedBy,
    IndexedByInstance,
    IndexedByPosition,
    InterpretedPeripheral,
    NodeChannels,
    ParsedDTNode,
    PeripheralResult,
    ProcessorInfo,
    dt_processor_info,
    sort_configs_by_type,
)

# List of Fields that should be given as 32-bit integers, regardless of bitfield width
FORCE_32_BIT_FIELDS = ["ADDRESS"]


# Special grouping of compatibles that should always be ignored as peripheral resources for the
# associated list of domains. Note that compatibles are checked with compatibles_contain,
# so suffixes can be obviated in this table.
IGNORED_PERIPH_COMPATIBLES = [
    (
        "nordic,nrf-bellboard",
        [
            DomainID.GLOBAL_FAST,
            DomainID.APPLICATION,
            DomainID.RADIOCORE,
        ],
    ),
    (
        "nordic,nrf-vevif",
        [
            DomainID.GLOBAL_FAST,
            DomainID.APPLICATION,
            DomainID.RADIOCORE,
        ],
    ),
    ("nordic,nrf-clic", [DomainID.GLOBAL_FAST]),
    (
        "nordic,nrf-gpio",
        [
            DomainID.GLOBAL_FAST,
            DomainID.APPLICATION,
            DomainID.RADIOCORE,
        ],
    ),
]


# Regex for memory region compatibles that should be allocated via UICR
MEM_COMPAT_REGEX = re.compile(r"^nordic,owned-(memory|partitions)$")

# Dimension of the UICR.GPIO[].PIN[] register
GPIO_PIN_CTRLSEL_PER_PORT = 14

# Collection of logs from extract functions
UICR_LOGS = defaultdict(list)


class DppiLogInfo(NamedTuple):
    """Helper tuple for DPPI logging information."""

    config: DppicConfig
    channels: NodeChannels
    mapped_indexing: bool
    separate_source_sink: bool


class IpcmapLogInfo(NamedTuple):
    """Helper tuple for IPCMAP logging information."""

    node: ParsedDTNode
    source_map: IpcMapConfig
    sink_map: IpcMapConfig


class PinctrlLogInfo(NamedTuple):
    """Helper tuple for Pinctrl logging information."""

    gpio: GpioPin
    config_node: Any
    group_node: Any


class GpiosLogInfo(NamedTuple):
    """Helper tuple for GPIO node logging information."""

    gpio: GpioPin
    ctrldata: edtlib.ControllerAndData
    cell_data: dict


class PinPropLogInfo(NamedTuple):
    """Helper tuple for peripheral pin property logging information."""

    gpio: GpioPin
    prop: edtlib.Property


class ConfigTypeEnum(enum.Enum):
    """Enumeration of different UICR config types."""

    VTOR = enum.auto()
    MEMORY = enum.auto()
    PERIPH = enum.auto()
    IPCT = enum.auto()
    IPCMAP = enum.auto()
    DPPI = enum.auto()
    GRTC = enum.auto()
    MAILBOX = enum.auto()
    GPIOTE = enum.auto()
    GPIO = enum.auto()
    PTREXTUICR = enum.auto()


class SvdEnum:
    """Helper class for constructing commonly used SVD enums."""

    ENABLED = "Enabled"
    DISABLED = "Disabled"

    @classmethod
    def enabled_if(cls, is_enabled: bool) -> str:
        return cls.ENABLED if is_enabled else cls.DISABLED

    ALLOWED = "Allowed"
    NOT_ALLOWED = "NotAllowed"

    @classmethod
    def allowed_if(cls, is_allowed: bool) -> str:
        return cls.ALLOWED if is_allowed else cls.NOT_ALLOWED

    SECURE = "Secure"
    NONSECURE = "NonSecure"

    @classmethod
    def secure_if(cls, is_secure: bool) -> str:
        return cls.SECURE if is_secure else cls.NONSECURE

    OWN = "Own"
    NOT_OWN = "NotOwn"

    SINK = "Sink"
    SOURCE = "Source"

    LINKED = "Linked"
    NOT_LINKED = "NotLinked"


def make_ch_fields(
    channels: Iterable[int], value: Union[str, int]
) -> Dict[str, Union[str, int]]:
    """Make a dictionary of CH_i field values for the given channels."""
    return _make_indexed_fields("CH", channels, value)


def make_cc_fields(
    ccs: Iterable[int], value: Union[str, int]
) -> Dict[str, Union[str, int]]:
    """Make a dictionary of CC_i field values for the given channels."""
    return _make_indexed_fields("CC", ccs, value)


def make_pin_fields(
    pins: Iterable[int], value: Union[str, int]
) -> Dict[str, Union[str, int]]:
    """Make a dictionary of PIN_i field values for the given channels."""
    return _make_indexed_fields("PIN", pins, value)


def _make_indexed_fields(
    field_base_name: str, indices: Iterable[int], value: Union[str, int]
) -> Dict[str, Union[str, int]]:
    return {f"{field_base_name}_{i}": value for i in indices}


@dataclass(frozen=True, order=True)
class GpioConfig(ExtractedConfig):
    """General purpose I/O (GPIO) configuration."""

    instance: ParsedDTNode
    owned_pins: List[int] = dc.field(default_factory=list)
    nonsecure_pins: List[int] = dc.field(default_factory=list)
    pin_ctrlsels: Dict[int, int] = dc.field(default_factory=dict)

    def __post_init__(self):
        self.owned_pins.sort()
        self.nonsecure_pins.sort()

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByInstance(instance=self.instance)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record({f"gpio_instance_{index}": {"ADDRESS": self.instance.address}})

        if self.owned_pins:
            record[f"gpio_own_{index}"] = make_pin_fields(self.owned_pins, SvdEnum.OWN)

        if self.nonsecure_pins:
            record[f"gpio_secure_{index}"] = make_pin_fields(
                self.nonsecure_pins, SvdEnum.NONSECURE
            )

        if self.pin_ctrlsels:
            for pin, ctrlsel in self.pin_ctrlsels.items():
                flat_index = index * GPIO_PIN_CTRLSEL_PER_PORT + pin
                record[f"gpio_pin_ctrlsel_{flat_index}"] = {"CTRLSEL": ctrlsel}

        return record


@dataclass(frozen=True, order=True)
class MemoryConfig(ExtractedConfig):
    """Memory configuration."""

    address: int
    size: int
    owner: OwnerID = OwnerID.SECURE
    readable: bool = False
    writable: bool = False
    executable: bool = False
    secure: bool = True
    non_secure_callable: bool = False

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByPosition(num_slots=1)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record(
            {
                f"mem_config_{index}": {
                    "ADDRESS": self.address,
                    "OWNERID": int(self.owner),
                    "READ": SvdEnum.allowed_if(self.readable),
                    "WRITE": SvdEnum.allowed_if(self.writable),
                    "EXECUTE": SvdEnum.allowed_if(self.executable),
                    "SECURE": SvdEnum.secure_if(self.secure),
                    "NSC": SvdEnum.enabled_if(self.non_secure_callable),
                },
                f"mem_size_{index}": {"SIZE": self.size},
            }
        )

        return record

    @property
    def end_address(self) -> int:
        return self.address + self.size

    def adjacent_or_overlap(self, other: MemoryConfig) -> bool:
        """
        Check if this configuration's address range overlaps with or is adjacent with other's
        address range.
        """
        return (
            self.address + self.size >= other.address
            and other.address + other.size >= self.address
        )

    def settings_equal(self, other: MemoryConfig) -> bool:
        """
        Check if this configuration has the same settings as other.
        Settings include all memory configuration properties except for address and size.
        """
        return (
            self.owner,
            self.readable,
            self.writable,
            self.executable,
            self.secure,
            self.non_secure_callable,
        ) == (
            other.owner,
            other.readable,
            other.writable,
            other.executable,
            other.secure,
            other.non_secure_callable,
        )


@dataclass(frozen=True, order=True)
class PeripheralConfig(ExtractedConfig):
    """Peripheral configuration."""

    instance: ParsedDTNode
    processor: ProcessorID = ProcessorID.SECURE
    secure: bool = True
    dma_secure: bool = True

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByPosition(num_slots=1)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record(
            {
                f"periph_config_{index}": {
                    "ADDRESS": self.instance.address,
                    "PROCESSOR": int(self.processor),
                    "SECURE": SvdEnum.secure_if(self.secure),
                    "DMASEC": SvdEnum.secure_if(self.dma_secure),
                }
            }
        )

        return record


@dataclass(frozen=True, order=True)
class IpcMapConfig(ExtractedConfig):
    """Inter-processor communication map (IPCMAP) configuration."""

    source_domain: DomainID
    source_channel: int
    sink_domain: DomainID
    sink_channel: int

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByPosition(num_slots=1)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record(
            {
                f"ipcmap_{index}": {
                    "DOMAINIDSOURCE": int(self.source_domain),
                    "IPCTCHSOURCE": self.source_channel,
                    "DOMAINIDSINK": int(self.sink_domain),
                    "IPCTCHSINK": self.sink_channel,
                }
            }
        )

        return record


@dataclass(frozen=True, order=True)
class IpctConfig(ExtractedConfig):
    """Interprocessor communication transceiver (IPCT) configuration."""

    instance: ParsedDTNode
    owned_channels: List[int] = dc.field(default_factory=list)
    nonsecure_channels: List[int] = dc.field(default_factory=list)

    def __post_init__(self):
        self.owned_channels.sort()
        self.nonsecure_channels.sort()

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByInstance(instance=self.instance)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record(
            {
                f"ipct_global_instance_{index}": {"ADDRESS": self.instance.address},
            }
        )

        if self.owned_channels:
            record[f"ipct_global_ch_own_{index}"] = make_ch_fields(
                self.owned_channels, SvdEnum.OWN
            )

        if self.nonsecure_channels:
            record[f"ipct_global_ch_secure_{index}"] = make_ch_fields(
                self.nonsecure_channels, SvdEnum.NONSECURE
            )

        return record


@dataclass(frozen=True, order=True)
class MailboxConfig(ExtractedConfig):
    """Mailbox configuration."""

    address: int
    size: int
    owner: OwnerID = OwnerID.SECURE
    secure: bool = True

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByPosition(num_slots=1)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record(
            {
                f"mailbox_address_{index}": {"ADDRESS": self.address},
                f"mailbox_config_{index}": {
                    "SECURE": SvdEnum.secure_if(self.secure),
                    "OWNERID": int(self.owner),
                    "SIZE": self.size,
                },
            }
        )

        return record


@dataclass(frozen=True, order=True)
class DualMailboxConfig(ExtractedConfig):
    """Split transmit/receive buffer mailbox configuration."""

    tx: MailboxConfig
    rx: MailboxConfig

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByPosition(
            num_slots=self.tx.indexed_by.num_slots + self.rx.indexed_by.num_slots
        )

    def to_record(self, *, index: int, **kwargs) -> Record:
        tx_record = self.tx.to_record(index=index)
        rx_record = self.rx.to_record(index=index + self.tx.indexed_by.num_slots)

        record = Record(
            chain(
                tx_record.items(),
                rx_record.items(),
            )
        )

        return record


@dataclass(frozen=True, order=True)
class DppicConfig(ExtractedConfig):
    """Distributed programmable peripheral interconnect controller (DPPIC) configuration."""

    instance: ParsedDTNode
    owned_channels: List[int] = dc.field(default_factory=list)
    nonsecure_channels: List[int] = dc.field(default_factory=list)
    source_channels: List[int] = dc.field(default_factory=list)
    sink_channels: List[int] = dc.field(default_factory=list)
    owned_channel_groups: List[int] = dc.field(default_factory=list)
    nonsecure_channel_groups: List[int] = dc.field(default_factory=list)

    def __post_init__(self):
        self.owned_channels.sort()
        self.nonsecure_channels.sort()
        self.source_channels.sort()
        self.sink_channels.sort()
        self.owned_channel_groups.sort()
        self.nonsecure_channel_groups.sort()

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByInstance(instance=self.instance)

    def to_record(
        self, *, index: int, separate_source_sink: bool = False, **kwargs
    ) -> Record:
        record = Record(
            {
                f"dppi_global_instance_{index}": {"ADDRESS": self.instance.address},
            }
        )

        if self.owned_channels:
            record[f"dppi_global_ch_own_{index}"] = make_ch_fields(
                self.owned_channels, SvdEnum.OWN
            )

        if self.nonsecure_channels:
            record[f"dppi_global_ch_secure_{index}"] = make_ch_fields(
                self.nonsecure_channels, SvdEnum.NONSECURE
            )

        if separate_source_sink:
            if self.source_channels:
                record[f"dppi_global_ch_link_source_{index}"] = make_ch_fields(
                    self.source_channels, SvdEnum.LINKED
                )

            if self.sink_channels:
                record[f"dppi_global_ch_link_sink_{index}"] = make_ch_fields(
                    self.sink_channels, SvdEnum.LINKED
                )
        else:
            record[f"dppi_global_ch_link_en_{index}"] = make_ch_fields(
                chain(self.source_channels, self.sink_channels), SvdEnum.ENABLED
            )
            record[f"dppi_global_ch_link_dir_{index}"] = make_ch_fields(
                self.sink_channels, SvdEnum.SINK
            )

        if self.owned_channel_groups:
            record[f"dppi_global_chg_own_{index}"] = _make_indexed_fields(
                "CHG", self.owned_channel_groups, SvdEnum.OWN
            )

        if self.nonsecure_channel_groups:
            record[f"dppi_global_chg_secure_{index}"] = _make_indexed_fields(
                "CHG", self.nonsecure_channel_groups, SvdEnum.NONSECURE
            )

        return record


@dataclass(frozen=True, order=True)
class GpioteConfig(ExtractedConfig):
    """GPIO tasks and events (GPIOTE) configuration."""

    instance: ParsedDTNode
    owned_channels: List[int] = dc.field(default_factory=list)
    nonsecure_channels: List[int] = dc.field(default_factory=list)

    def __post_init__(self):
        self.owned_channels.sort()
        self.nonsecure_channels.sort()

    @property
    def indexed_by(self) -> IndexedBy:
        return IndexedByInstance(instance=self.instance)

    def to_record(self, *, index: int, **kwargs) -> Record:
        record = Record(
            {
                f"gpiote_instance_{index}": {"ADDRESS": self.instance.address},
                f"gpiote_ch_own_{index}": make_ch_fields(
                    self.owned_channels, SvdEnum.OWN
                ),
            }
        )
        if self.nonsecure_channels:
            record[f"gpiote_ch_secure_{index}"] = make_ch_fields(
                self.nonsecure_channels, SvdEnum.NONSECURE
            )

        return record


@dataclass(frozen=True, order=True)
class GrtcConfig(ExtractedConfig):
    """Global real-time counter (GRTC) configuration."""

    owned_channels: List[int] = dc.field(default_factory=list)
    nonsecure_channels: List[int] = dc.field(default_factory=list)

    def __post_init__(self):
        self.owned_channels.sort()
        self.nonsecure_channels.sort()

    def to_record(self, **kwargs) -> Record:
        record = Record(
            {"grtc_cc_own_0": make_cc_fields(self.owned_channels, SvdEnum.OWN)}
        )

        if self.nonsecure_channels:
            record["grtc_cc_secure_0"] = make_cc_fields(
                self.nonsecure_channels, SvdEnum.NONSECURE
            )

        return record


@dataclass(frozen=True, order=True)
class SecureVtorConfig(ExtractedConfig):
    """Secure vector table configuration."""

    initial_vtor: int

    def to_record(self, **kwargs) -> Record:
        record = Record({"initsvtor_0": {"INITSVTOR": self.initial_vtor}})

        return record


@dataclass(frozen=True, order=True)
class NonSecureVtorConfig(ExtractedConfig):
    """Non-secure vector table configuration."""

    initial_vtor: int

    def to_record(self, **kwargs) -> Record:
        record = Record({"initsvtor_0": {"INITSVTOR": self.initial_vtor}})

        return record


@dataclass(frozen=True, order=True)
class PtrExtUicrConfig(ExtractedConfig):
    """Extended UICR pointer configuration."""

    address: int

    def to_record(self, **kwargs) -> Record:
        record = Record({"ptrextuicr_0": {"PTREXTUICR": self.address}})

        return record


class GpioPin(NamedTuple):
    """
    Helper tuple for pin-related information.
    """

    port: int
    pin: int
    secure: bool
    ctrlsel: Optional[int]


@enum.unique
class SourceMapIndex(enum.IntEnum):
    """
    Enumeration of indices in the IPCT array property that defines channel/domain info for when the
    owning domain is a source.
    """

    SOURCE_CH = 0
    SINK_DOMAIN = enum.auto()
    SINK_CH = enum.auto()


@enum.unique
class SinkMapIndex(enum.IntEnum):
    """
    Enumeration of indices in the IPCT array property that defines channel/domain info for when the
    owning domain is a sink.
    """

    SINK_CH = 0
    SOURCE_DOMAIN = enum.auto()
    SOURCE_CH = enum.auto()


class NodeType(enum.Enum):
    """
    Enumeration of node types that are relevant to UICR.
    """

    MEMORY = enum.auto()
    PERIPH = enum.auto()
    GPIOTE = enum.auto()
    GPIO = enum.auto()
    GRTC = enum.auto()
    GLOBAL_DPPIC = enum.auto()
    GLOBAL_IPCT = enum.auto()
    LOCAL_IPCT = enum.auto()
    MAILBOX = enum.auto()


class ResourceCompatible(enum.Enum):
    """
    Enumeration of devicetree compatibles that are of relevant to nodes containing information from
    UICR. These can be base compatible substring that are not version specific in order to allow
    for more fuzzy checking of relevant nodes, preventing breakage from naming convention changes
    in the external repository that defines them.
    """

    GPIO = "nordic,nrf-gpio"
    GPIOTE = "nordic,nrf-gpiote"
    GRTC = "nordic,nrf-grtc"
    DPPIC = "nordic,nrf-dppic"
    IPCT = "nordic,nrf-ipct"
    CLIC = "nordic,nrf-clic"
    MAILBOX_ICMSG = "zephyr,ipc-icmsg"
    MAILBOX_RPMSG = "zephyr,ipc-openamp-static-vrings"


class InterpretedUICR(InterpretedPeripheral):
    """
    UICR-related information interpreted from the parsed devicetree nodes that are relevant to the
    peripheral's configurations.
    """

    def __init__(self, devicetree: edtlib.EDT):
        """
        Initialize the class attribute(s).

        :param devicetree: edtlib Devicetree object
        """
        super().__init__(devicetree=devicetree, name="UICR")

        self._processor: ProcessorInfo = dt_processor_info(devicetree)
        self._resource_nodes = defaultdict(list)
        self._gpios: List[edtlib.Property] = []
        self._pinctrls: List[edtlib.PinCtrl] = []
        self._gpio_pins: List[edtlib.Property] = []
        self._uicr_node: Optional[ParsedDTNode] = None

        has_relevant_status = lambda n: (n.enabled or n.reserved)
        has_relevant_address = lambda n: (
            n.domain == DomainID.GLOBAL
            and n.region in (AddressRegion.PERIPHERAL, AddressRegion.STM)
        )
        has_relevant_periph_compatible = lambda n: (
            n.compatibles_contain("nordic")
            and not any(
                n.compatibles_contain(ignored_compat)
                and self.domain_id in ignored_domains
                for ignored_compat, ignored_domains in IGNORED_PERIPH_COMPATIBLES
            )
        )

        for node in (ParsedDTNode(n) for n in devicetree.nodes):
            if node.compatibles_contain("nordic,nrf-uicr"):
                domain_id = node.properties.get_val("domain")
                if domain_id == self.domain_id.value:
                    if self._uicr_node is not None:
                        raise RuntimeError(
                            f"Duplicate domain ID ({self.domain_id.value}) "
                            f"found in {self._uicr_node.path} and {node.path}"
                        )
                    self._uicr_node = node
                elif domain_id is not None:
                    continue
                self._properties.update(node.properties)
            elif has_relevant_status(node) and node.compatibles_match(MEM_COMPAT_REGEX):
                self._resource_nodes[NodeType.MEMORY].append(node)
            elif (
                has_relevant_address(node)
                and ResourceCompatible.GPIO.value in node.compatibles
            ):
                self._resource_nodes[NodeType.GPIO].append(node)
            elif has_relevant_status(node) and has_relevant_address(node):
                if node.compatibles_contain(ResourceCompatible.GPIOTE.value):
                    self._resource_nodes[NodeType.GPIOTE].append(node)
                elif node.compatibles_contain(ResourceCompatible.GRTC.value):
                    self._resource_nodes[NodeType.GRTC].append(node)
                elif node.compatibles_contain(ResourceCompatible.DPPIC.value):
                    self._resource_nodes[NodeType.GLOBAL_DPPIC].append(node)
                elif node.compatibles_contain(ResourceCompatible.IPCT.value):
                    self._resource_nodes[NodeType.GLOBAL_IPCT].append(node)
                elif has_relevant_periph_compatible(node):
                    self._resource_nodes[NodeType.PERIPH].append(node)
            elif has_relevant_status(node) and node.domain == self.domain_id:
                if node.compatibles_contain(ResourceCompatible.IPCT.value):
                    self._resource_nodes[NodeType.LOCAL_IPCT].append(node)
            elif has_relevant_status(node) and (
                node.compatibles_contain(ResourceCompatible.MAILBOX_ICMSG.value)
                or node.compatibles_contain(ResourceCompatible.MAILBOX_RPMSG.value)
            ):
                self._resource_nodes[NodeType.MAILBOX].append(node)

            if has_relevant_status(node) or "status" not in node.properties:
                gpios_props = (
                    prop
                    for prop in node.properties.values()
                    if _is_gpios_prop(prop)
                    and any((_has_nordic_gpio_controller(v) for v in prop.val))
                )
                self._gpios.extend(gpios_props)
            if has_relevant_status(node) and node.compatibles_contain("nordic"):
                self._pinctrls.extend(node.pinctrls)

                gpio_pin_props = [
                    p for p in node.properties.values() if p.name.endswith("-pin")
                ]
                if gpio_pin_props:
                    self._gpio_pins.extend(gpio_pin_props)

    @cached_property
    def uicr_ext_node(self) -> ParsedDTNode:
        """
        DT node representing the local domain's extended UICR.
        """
        node = self.properties.get_val("ptr-ext-uicr")
        if node is None:
            raise RuntimeError(
                "No extended UICR node was found in the devicetree. "
                "Expected a 'ptr-ext-uicr' property of UICR to point to the node."
            )

        return ParsedDTNode(node)

    @cached_property
    def vtor_node(self) -> ParsedDTNode:
        """
        DT node representing the local domain's VTOR or code partition.
        """
        node = self.chosen["zephyr,code-partition"]
        if node is None:
            raise RuntimeError(
                "No VTOR node was found in the devicetree. "
                "Expected a 'zephyr,code-partition' chosen property to point to the node."
            )
        parent = node.parent
        if not parent or "nordic,owned-partitions" not in parent.compatibles:
            raise RuntimeError(
                f"Expected VTOR node {node.path} to be a partition node. "
                "Its parent node does not have a 'nordic,owned-partitions' compatible."
            )
        return node

    @property
    def has_svtor(self) -> bool:
        """
        Returns true if the devicetree contains a secure VTOR.
        """
        return self.vtor_node.parent.properties.get_val("perm-secure", False)

    @property
    def svtor_address(self) -> Optional[int]:
        """
        Address to use for INITSVTOR.
        """
        return self.vtor_node.address if self.has_svtor else None

    @property
    def nsvtor_address(self) -> Optional[int]:
        """
        Address to use for INITNSVTOR.
        """
        return None if self.has_svtor else self.vtor_node.address

    @property
    def address(self) -> int:
        """
        Base address of the UICR as defined by the devicetree.
        """
        uicr_node = self._uicr_node
        if uicr_node is None:
            raise RuntimeError(
                "No UICR node was found in the devicetree. Expected a 'nordic,nrf-uicr-v2' "
                f"compatible node with 'domain' {self.domain_id.value}."
            )
        address = uicr_node.address
        if address is None:
            raise RuntimeError(f"Missing 'reg' property in UICR node {uicr_node}")
        return address

    @property
    def processor(self) -> ProcessorInfo:
        """
        Information of the UICR's related processor.
        """
        return self._processor

    @cached_property
    def domain_id(self) -> DomainID:
        """
        ID of the UICR's related domain.
        """

        domain_id = DomainID.from_processor(self.processor.cpu_id)
        if domain_id is None:
            raise RuntimeError(
                f"Unexpected Domain ID = {domain_id!r} value for CPU ID {self.processor.cpu_id!r}"
            )
        return domain_id

    @property
    def periph_nodes(self) -> List[ParsedDTNode]:
        """
        Global peripherals in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.PERIPH]

    @property
    def memory_nodes(self) -> List[ParsedDTNode]:
        """
        Memory nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.MEMORY]

    @property
    def gpiote_nodes(self) -> List[ParsedDTNode]:
        """
        GPIOTE nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.GPIOTE]

    @property
    def gpio_nodes(self) -> List[ParsedDTNode]:
        """
        GPIO nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.GPIO]

    @property
    def grtc_nodes(self) -> List[ParsedDTNode]:
        """
        GRTC nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.GRTC]

    @property
    def global_dppic_nodes(self) -> List[ParsedDTNode]:
        """
        Global DPPIC nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.GLOBAL_DPPIC]

    @property
    def global_ipct_nodes(self) -> List[ParsedDTNode]:
        """
        Global IPCT nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.GLOBAL_IPCT]

    @property
    def local_ipct_nodes(self) -> List[ParsedDTNode]:
        """
        Local IPCT nodes in the compiled devicetree that contain relevant properties for UICR.
        These are only necessary for IPCMAP construction, and are not allocated as UICR resources themselves.
        """
        return self._resource_nodes[NodeType.LOCAL_IPCT]

    @property
    def mailbox_nodes(self) -> List[ParsedDTNode]:
        """
        MAILBOX nodes in the compiled devicetree that contain relevant properties for UICR.
        """
        return self._resource_nodes[NodeType.MAILBOX]

    @property
    def gpio_pin_props(self) -> List[edtlib.Property]:
        """
        GPIO pin properties that are relevant for UICR.
        """
        return self._gpio_pins

    @property
    def gpios(self) -> List[edtlib.Property]:
        """
        'gpios' property entries referencing GPIO pins that are relevant for UICR.
        """
        return self._gpios

    @property
    def pinctrls(self) -> List[edtlib.PinCtrl]:
        """
        Pin Control objects referencing GPIO pins that are relevant for UICR.
        """
        return self._pinctrls

    @property
    def mapped_instance_indexing(self) -> bool:
        """
        Returns true if the UICR uses mapped instance indexing for some resources, rather than pure
        lists through the presence of INSTANCE registers.
        """
        return self.properties.get_val("mapped-instance-indexing", False)

    def __str__(self) -> str:
        """String representation of the class."""

        props = {
            "Properties": [str(v) for v in self.properties.values()],
            "Chosen": self.chosen,
        }
        nodes = {k.name: [str(n) for n in v] for k, v in self._resource_nodes.items()}
        return (
            f"<UICR@0x{self.properties.get_val('reg')[0]:08x}, Domain {self.domain_id} - {str(self.processor)}>\n"
            f"{pformat(props)}\n{pformat(nodes)}"
        )


def from_dts(devicetree: edtlib.EDT, use_pinctrl: bool = True) -> PeripheralResult:
    """
    Gather information from parsed devicetree nodes on configurations that must set in UICR
    in order to be allocated on boot, then map those records into a memory map of UICR
    contents.

    :param devicetree: Devicetree structure.
    :param use_pinctrl: Use pinctrl for extracting GPIO records from peripheral nodes.

    :return: Meta-record of extracted UICR records from devicetree nodes.
    """

    uicr = InterpretedUICR(devicetree)
    log_dev(f"\n{str(uicr)}\n")

    address: int = uicr.address

    configs: List[ExtractedConfig] = extract_all_configs(uicr, use_pinctrl=use_pinctrl)

    records: List[Record] = convert_configs_to_records(configs, uicr)

    if verbose_logs():
        _log_extracted_uicr_configs(uicr)

    log_dev(f"\nAll interpreted DTS records:\n{pformat(records)}\n\n")

    uicr_record = Record()
    for record in [x for x in records if x]:
        uicr_record.update(record)

    return PeripheralResult(base_address=address, record=uicr_record)


def extract_all_configs(
    uicr: InterpretedUICR, use_pinctrl: bool = True
) -> List[ExtractedConfig]:
    """
    Extract all UICR configurations.
    """
    configs: List[ExtractedConfig] = []

    configs.extend(extract_memory_configs(uicr.memory_nodes, uicr.domain_id))

    configs.extend(extract_vtor_configs(uicr.svtor_address, uicr.nsvtor_address))

    configs.extend(extract_periph_configs(uicr.periph_nodes, uicr.processor))

    configs.extend(
        extract_dppic_configs(
            uicr.global_dppic_nodes,
            separate_source_sink=uicr.properties.get_val(
                "dppic-separate-source-sink", False
            ),
            mapped_indexing=uicr.mapped_instance_indexing,
            secure=uicr.has_svtor,
        )
    )

    configs.extend(
        extract_ipct_configs(
            uicr.global_ipct_nodes,
            secure=uicr.has_svtor,
        )
    )

    configs.extend(extract_ipcmap_configs(uicr.global_ipct_nodes))

    configs.extend(extract_ipcmap_configs(uicr.local_ipct_nodes))

    configs.extend(extract_mailbox_configs(uicr.mailbox_nodes))

    configs.extend(
        extract_grtc_configs(
            uicr.grtc_nodes,
            secure=uicr.has_svtor,
        )
    )

    configs.extend(
        extract_gpiote_configs(
            uicr.gpiote_nodes,
            secure=uicr.has_svtor,
        )
    )

    configs.append(extract_ptrextuicr_config(uicr.uicr_ext_node))

    if use_pinctrl:
        pinctrls = uicr.pinctrls
        gpio_pin_props = []
    else:
        pinctrls = []
        gpio_pin_props = uicr.gpio_pin_props

    configs.extend(
        extract_gpio_configs(
            uicr.gpio_nodes,
            uicr.gpios,
            pinctrls=pinctrls,
            gpio_pin_props=gpio_pin_props,
        )
    )

    return configs


def convert_configs_to_records(
    configs: List[ExtractedConfig], uicr: InterpretedUICR
) -> List[Record]:
    """
    Convert a list of extracted configurations to a list of UICR peripheral records.
    The records are sorted and indexed automatically.

    :param uicr: Interpreted UICR.
    :param configs: List of extracted configurations.

    :return: List of UICR peripheral records.
    """

    if uicr.mapped_instance_indexing:
        # Use fixed instance indexing for each type of config
        instance_map: Dict[type, Dict[ParsedDTNode, int]] = {
            config_type: {
                instance: index
                for index, instance in enumerate(map(ParsedDTNode, instances))
            }
            for config_type, instances in [
                (GpioConfig, uicr.properties.get_val("gpio-ports", [])),
                (GpioteConfig, uicr.properties.get_val("gpiote", [])),
                (DppicConfig, uicr.properties.get_val("global-dppic", [])),
                (IpctConfig, uicr.properties.get_val("global-ipct", [])),
            ]
        }
    else:
        # Dynamically map instances to indices
        instance_map: Dict[type, Dict[ParsedDTNode, int]] = defaultdict(dict)

    records: List[Record] = []

    # Index counter for array indexed configurations
    index_counter = defaultdict(int)

    # Additional keyword arguments to pass to to_record() methods
    extra_record_kwargs = {
        "separate_source_sink": uicr.properties.get_val(
            "dppic-separate-source-sink", False
        )
    }

    for config_list in sort_configs_by_type(configs):
        for config in config_list:
            indexed_by: Optional[IndexedBy] = config.indexed_by
            if isinstance(indexed_by, IndexedByInstance):
                instance: ParsedDTNode = indexed_by.instance

                if uicr.mapped_instance_indexing:
                    try:
                        config_index = instance_map[type(config)][instance]
                    except KeyError:
                        raise RuntimeError(
                            f"Unable to map node '{instance.path}' to the index of a known "
                            "global resource node configurable by UICR."
                        )
                else:  # Use automatic instance indexing
                    config_index = instance_map[type(config)].get(instance)
                    if config_index is None:  # New instance; add it to the map
                        config_index = index_counter[type(config)]
                        index_counter[type(config)] += 1
                        instance_map[type(config)][instance] = config_index

                record = config.to_record(index=config_index, **extra_record_kwargs)

            elif isinstance(indexed_by, IndexedByPosition):
                config_index = index_counter[type(config)]
                index_counter[type(config)] += indexed_by.num_slots

                record = config.to_record(index=config_index, **extra_record_kwargs)

            else:
                record = config.to_record(**extra_record_kwargs)

            records.append(record)

    return records


def extract_vtor_configs(
    svtor_address: Optional[int], nsvtor_address: Optional[int]
) -> List[Union[SecureVtorConfig, NonSecureVtorConfig]]:
    """
    Extract INITSVTOR and INITNSVTOR record information from parsed devicetree nodes.

    :param chosen: Devicetree node containing chosen properties

    :return: List of VTOR records for use with UICR.
    """

    vtors: List[Union[SecureVtorConfig, NonSecureVtorConfig]] = []

    if svtor_address is not None:
        config = SecureVtorConfig(initial_vtor=svtor_address)
        vtors.append(config)

        _append_log_config(ConfigTypeEnum.VTOR, config)

    if nsvtor_address is not None:
        config = NonSecureVtorConfig(initial_vtor=nsvtor_address)
        vtors.append(config)

        _append_log_config(ConfigTypeEnum.VTOR, config)

    return vtors


def extract_memory_configs(
    nodes: List[ParsedDTNode],
    domain_id: DomainID,
) -> List[MemoryConfig]:
    """
    Extract memory configuration record information from a list of parsed devicetree nodes.

    :param nodes: Parsed devicetree nodes
    :param domain_id: Domain ID associated to the UICR processor.

    :return: List of memory records for use with UICR.
    """

    default_owner_id_val: int = OwnerID.from_domain(domain_id).value

    unmerged_configs: List[Tuple[MemoryConfig, ParsedDTNode]] = []

    for node in nodes:
        if not node.compatibles_match(PARTITION_COMPAT_REGEX):
            addr = node.address
            size = node.length
        else:
            partitions = tuple(map(ParsedDTNode, node.node.children.values()))
            if not partitions:
                continue
            addr = min(p.address for p in partitions)
            size = max(p.address + p.length for p in partitions) - addr

        config = MemoryConfig(
            address=addr,
            size=size,
            owner=OwnerID.from_value(
                node.properties.get_val("owner-id", default_owner_id_val)
            ),
            readable=node.properties.get_val("perm-read", False),
            writable=node.properties.get_val("perm-write", False),
            executable=node.properties.get_val("perm-execute", False),
            secure=node.properties.get_val("perm-secure", False),
            non_secure_callable=node.properties.get_val("non-secure-callable", False),
        )

        unmerged_configs.append((config, node))

    merged_configs: List[Tuple[MemoryConfig, List[ParsedDTNode]]] = []

    # Bucket the configs based on overlapping address range and equality
    for config, node in unmerged_configs:
        for i, (m_config, m_nodes) in enumerate(merged_configs):
            if m_config.adjacent_or_overlap(config) and m_config.settings_equal(config):
                address = min(m_config.address, config.address)
                size = max(m_config.end_address, config.end_address) - address
                new_m_config = dc.replace(m_config, address=address, size=size)
                merged_configs[i] = (new_m_config, m_nodes + [node])
                break
        else:
            merged_configs.append((config, [node]))

    for m_config, m_nodes in merged_configs:
        _append_log_config(ConfigTypeEnum.MEMORY, (m_nodes, m_config))

    return [m_config for m_config, _ in merged_configs]


def _processor_id_from_node(node: ParsedDTNode, default: ProcessorID) -> ProcessorID:
    """
    Determine the processor ID of the CPU that will receive IRQs for a given peripheral.
    This is mapped from the interrupt controller associated with the peripheral node.

    :param node: Parsed devicetree node
    :param default: Processor ID of the domain CPU, which is used by default
    """

    current_processor_id = None

    for irq in node.node.interrupts:
        irq_ctrl = ParsedDTNode(irq.controller)
        processor_id = ProcessorID.from_nodelabels(irq_ctrl.labels)
        if processor_id is None:
            processor_id = default

        if current_processor_id is None:
            current_processor_id = processor_id
        elif current_processor_id != processor_id:
            raise RuntimeError(
                f"Peripheral node {node.path} has IRQs mapped to multiple processors "
                f"({current_processor_id} and {processor_id}), which is not supported."
            )

    return current_processor_id or default


def extract_periph_configs(
    nodes: List[ParsedDTNode], processor: ProcessorInfo
) -> List[PeripheralConfig]:
    """
    Extract peripheral config record information from a list of parsed devicetree nodes. If the
    peripheral contains GPIO pin properties, records for these are extracted from those properties
    while parsing the node.

    :note: This is only relevant for global peripheral nodes, and should not be used for locals.

    :param nodes: Parsed devicetree nodes
    :param processor: Information related to the domain processor.

    :return: List of peripheral records for use with UICR.
    """

    configs: List[PeripheralConfig] = []

    for node in nodes:
        config = PeripheralConfig(
            instance=node,
            processor=_processor_id_from_node(node, default=processor.cpu_id),
            secure=node.secure,
            dma_secure=node.secure,
        )

        configs.append(config)

        _append_log_config(ConfigTypeEnum.PERIPH, (node.reserved, config))

    return configs


def _is_gpios_prop(prop: edtlib.Property) -> bool:
    """Returns True if prop is a standard property referencing gpio pins"""
    return prop.type == "phandle-array" and re.match(r"^(.*-)?gpios", prop.name)


def _has_nordic_gpio_controller(gpios_entry: edtlib.ControllerAndData) -> bool:
    """Returns True if gpios_entry refers to a nordic GPIO node"""
    return ResourceCompatible.GPIO.value in gpios_entry.controller.compats


def _gpio_pin_security_from_node(node: edtlib.Node) -> bool:
    """Get the security level of GPIO pin based on the security of the node using it"""
    node_secure = ParsedDTNode(node).secure
    return node_secure if node_secure is not None else True


def _extract_gpios_prop_pin(
    gpios_prop: edtlib.Property,
    gpios_entry: edtlib.ControllerAndData,
) -> GpioPin:
    """
    Get the set of GPIO pins present in the given *-gpios property.

    :param gpios_entry: an entry in a 'gpios' property
    :return: A (port, pin) tuple representing the GPIO pin used in the property
    """

    cell_data = {
        "controller": {"port": gpios_entry.controller.props["port"].val},
        "data": dict(gpios_entry.data),
    }

    port: int = cell_data["controller"]["port"]
    pin: int = cell_data["data"]["pin"]
    secure = _gpio_pin_security_from_node(gpios_entry.node)
    ctrlsel = dt_lookup_ctrlsel(gpios_prop, (port, pin))
    gpio_pin = GpioPin(port=port, pin=pin, secure=secure, ctrlsel=ctrlsel)

    _append_log_config(
        ConfigTypeEnum.GPIO, GpiosLogInfo(gpio_pin, gpios_entry, cell_data)
    )

    return gpio_pin


def _extract_pinctrl_prop_pins(pinctrl: edtlib.PinCtrl) -> Set[GpioPin]:
    """
    Get the set of GPIO pins used by one or more pinctrl configurations in the pinctrl node.

    :param pinctrl: Pin Control object
    :return: A set of GpioPins representing the GPIO pins used in at least one pinctrl config
    """

    pins: Set[GpioPin] = set()

    secure = _gpio_pin_security_from_node(pinctrl.node)

    for config_node in pinctrl.conf_nodes:
        for group_node in config_node.children.values():
            for psel_val in group_node.props["psels"].val:
                psel = nrf_psel_decode(psel_val)
                ctrlsel = dt_lookup_ctrlsel(pinctrl, psel)
                gpio_pin = GpioPin(
                    port=psel.port, pin=psel.pin, secure=secure, ctrlsel=ctrlsel
                )
                pins.add(gpio_pin)

                _append_log_config(
                    ConfigTypeEnum.GPIO,
                    PinctrlLogInfo(gpio_pin, config_node, group_node),
                )

    return pins


def _gpio_pins_from_props(
    gpios: List[edtlib.Property],
    pinctrls: List[edtlib.PinCtrl],
    gpio_pin_props: List[edtlib.Property],
) -> Set[GpioPin]:
    """
    Extract a list of GPIO pins from devicetree objects containing GPIO pin usage information.

    :param gpios: 'gpios' property entries
    :param pinctrls: Pin Control objects
    :param gpio_pin_props: List of GPIO pin properties

    :return: List of GPIO pins used by the given devicetree objects
    """

    pins: Set[GpioPin] = set()

    for gpios_prop in gpios:
        for gpios_entry in gpios_prop.val:
            if _has_nordic_gpio_controller(gpios_entry):
                gpio_pin = _extract_gpios_prop_pin(gpios_prop, gpios_entry)
                pins.add(gpio_pin)
            else:
                print(f"Mismatch on controller of: {gpios_entry}: {gpios_entry.controller.compats}")

    for pinctrl in pinctrls:
        pinctrl_pins = _extract_pinctrl_prop_pins(pinctrl)
        pins.update(pinctrl_pins)

    for prop in gpio_pin_props:
        prop_port, prop_pin = gpio_port_pin_decode(prop.val)
        secure = _gpio_pin_security_from_node(prop.node)
        gpio_pin = GpioPin(
            port=prop_port, pin=prop_pin, secure=secure, ctrlsel=None
        )
        pins.add(gpio_pin)

        _append_log_config(ConfigTypeEnum.GPIO, PinPropLogInfo(gpio_pin, prop))

    return pins


def extract_gpio_configs(
    nodes: List[ParsedDTNode],
    gpios: List[edtlib.Property],
    pinctrls: List[edtlib.PinCtrl],
    gpio_pin_props: List[edtlib.Property] = [],
) -> List[GpioConfig]:
    """
    Extract GPIO record information from parsed devicetree properties.

    :param nodes: Parsed devicetree nodes
    :param gpios: 'gpios' property entries
    :param pinctrls: Pin Control objects
    :param gpio_pin_props: GPIO pin properties

    :return: List of GPIO records for UICR.
    """

    configs: List[GpioConfig] = []

    if not (gpios or pinctrls or gpio_pin_props):
        return []

    port_to_instance = {n.properties["port"].val: n for n in nodes}
    gpio_pins = _gpio_pins_from_props(gpios, pinctrls, gpio_pin_props)

    get_port = lambda pin: pin.port
    gpio_pins_by_port = groupby(sorted(set(gpio_pins), key=get_port), key=get_port)

    for port, gpio_pins in gpio_pins_by_port:
        try:
            instance = port_to_instance[port]
        except KeyError:
            raise RuntimeError(
                f"GPIO port {port=} is not in supported ports ({port_to_instance.keys()})"
            )

        owned_pins: List[int] = []
        ns_pins: List[int] = []
        pin_ctrlsels: Dict[int, int] = {}

        for pin in sorted(gpio_pins, key=lambda p: p.pin):
            owned_pins.append(pin.pin)
            if pin.ctrlsel is not None:
                pin_ctrlsels[pin.pin] = pin.ctrlsel
            if not pin.secure:
                ns_pins.append(pin.pin)

        config = GpioConfig(
            instance=instance,
            owned_pins=owned_pins,
            nonsecure_pins=ns_pins,
            pin_ctrlsels=pin_ctrlsels,
        )

        configs.append(config)

    return configs


def extract_gpiote_configs(
    nodes: List[ParsedDTNode],
    secure: bool,
) -> List[GpioteConfig]:
    """
    Extract GPIOTE record information from a list of parsed devicetree nodes.

    :param nodes: Parsed devicetree nodes
    :param secure: Configure for a Trustzone secure system

    :return: List of GPIOTE configuration records for use with UICR.
    """

    configs: List[GpioteConfig] = []

    if not nodes:
        return []

    for node in nodes:
        channels = NodeChannels(
            owned=node.properties.get_val("owned-channels", []),
            child=node.properties.get_val("child-owned-channels", []),
            nonsecure=node.properties.get_val("nonsecure-channels", []),
        )

        if secure and node.secure:
            ns_channels: List[int] = channels.nonsecure
        else:
            ns_channels = copy(channels.owned)

        config = GpioteConfig(
            instance=node,
            owned_channels=channels.owned,
            nonsecure_channels=ns_channels,
        )

        configs.append(config)

        _append_log_config(ConfigTypeEnum.GPIOTE, (channels, config))

    return configs


def extract_grtc_configs(
    nodes: List[ParsedDTNode],
    secure: bool,
) -> List[GrtcConfig]:
    """
    Extract GRTC record information from a list of parsed devicetree nodes.

    :param nodes: Parsed devicetree nodes
    :param secure: Configure for a Trustzone secure system

    :return: List of GRTC configuration records for use with UICR.
    """

    configs: List[GrtcConfig] = []

    if not nodes:
        return []

    for node in nodes:
        channels = NodeChannels(
            owned=node.properties.get_val("owned-channels", []),
            child=node.properties.get_val("child-owned-channels", []),
            nonsecure=node.properties.get_val("nonsecure-channels", []),
        )

        if secure and node.secure:
            ns_channels: List[int] = channels.nonsecure
        else:
            ns_channels = copy(channels.owned)

        config = GrtcConfig(
            owned_channels=channels.owned,
            nonsecure_channels=ns_channels,
        )

        configs.append(config)

        _append_log_config(ConfigTypeEnum.GRTC, channels)

    return configs


def extract_dppic_configs(
    nodes: List[ParsedDTNode],
    mapped_indexing: bool,
    separate_source_sink: bool,
    secure: bool,
) -> List[DppicConfig]:
    """
    Extract DPPIC record information from a list of parsed devicetree nodes.

    :param nodes: Parsed devicetree nodes
    :param mapped_indexing:  UICR maps index n of UICR.DPPIC[n] against a specific DPPIC instance.
    :param separate_source_sink: UICR contains separate registers for SOURCE/SINK configurations.
    :param secure: Configure for a Trustzone secure system

    :return: List of DPPIC configuration records for use with UICR.
    """

    configs: List[DppicConfig] = []

    for node in nodes:
        channels = NodeChannels(
            owned=node.properties.get_val("owned-channels", []),
            child=node.properties.get_val("child-owned-channels", []),
            nonsecure=node.properties.get_val("nonsecure-channels", []),
            source=node.properties.get_val("source-channels", []),
            sink=node.properties.get_val("sink-channels", []),
        )

        owned_chg: List[int] = node.properties.get_val("owned-channel-groups", [])

        if secure and node.secure:
            ns_channels: List[int] = channels.nonsecure
            ns_chg: List[int] = node.properties.get_val("nonsecure-channel-groups", [])
        else:
            ns_channels = copy(channels.owned)
            ns_chg = copy(owned_chg)

        config = DppicConfig(
            instance=node,
            owned_channels=channels.owned,
            nonsecure_channels=ns_channels,
            source_channels=channels.source,
            sink_channels=channels.sink,
            owned_channel_groups=owned_chg,
            nonsecure_channel_groups=ns_chg,
        )

        configs.append(config)

        _append_log_config(
            ConfigTypeEnum.DPPI,
            DppiLogInfo(config, channels, mapped_indexing, separate_source_sink),
        )

    return configs


def extract_ipcmap_configs(nodes: List[ParsedDTNode]) -> List[IpcMapConfig]:
    """
    Extract IPCMAP record information from a list of IPCT nodes and returns a list of records
    to use with the other UICR resource records. Due to limitations of devicetree, this function
    needs to know what a "unit" of an IPC mapping looks like in the property, in other words a
    subset of an integer array comprising of the relevant information. See the description of the
    related IPCT properties for more information.

    :param nodes: List of parsed IPCT nodes from devicetree

    :return: List of IPCMAP records extracted from the source/sink mappings of the IPCT nodes.
    """

    configs: List[IpcMapConfig] = []

    for node in nodes:
        raw_source_map: List[int] = node.properties.get_val("source-channel-links", [])
        raw_sink_map: List[int] = node.properties.get_val("sink-channel-links", [])

        domain_id: DomainID = DomainID.from_value(
            node.properties.get_val("global-domain-id", node.domain.value)
        )

        log_dev(f"IPCMAP for {node.label_string()}")

        source_mapping = (
            [
                IpcMapConfig(
                    source_domain=domain_id,
                    source_channel=raw_source_map[i + SourceMapIndex.SOURCE_CH],
                    sink_domain=DomainID.from_value(
                        raw_source_map[i + SourceMapIndex.SINK_DOMAIN]
                    ),
                    sink_channel=raw_source_map[i + SourceMapIndex.SINK_CH],
                )
                for i in range(0, len(raw_source_map), len(SourceMapIndex))
            ]
            if raw_source_map
            else []
        )

        log_dev(pformat(raw_source_map))
        log_dev(pformat(source_mapping))

        sink_mapping = (
            [
                IpcMapConfig(
                    sink_domain=domain_id,
                    sink_channel=raw_sink_map[i + SinkMapIndex.SINK_CH],
                    source_domain=DomainID.from_value(
                        raw_sink_map[i + SinkMapIndex.SOURCE_DOMAIN]
                    ),
                    source_channel=raw_sink_map[i + SinkMapIndex.SOURCE_CH],
                )
                for i in range(0, len(raw_sink_map), len(SinkMapIndex))
            ]
            if raw_sink_map
            else []
        )

        log_dev(pformat(raw_sink_map))
        log_dev(pformat(sink_mapping))

        configs.extend(source_mapping)
        configs.extend(sink_mapping)

        _append_log_config(
            ConfigTypeEnum.IPCMAP, IpcmapLogInfo(node, source_mapping, sink_mapping)
        )

        log_dev("")

    return configs


def extract_ipct_configs(
    nodes: List[ParsedDTNode],
    secure: bool,
) -> List[IpctConfig]:
    """
    Extract IPCT channel record information from a list of parsed devicetree nodes.

    :param nodes: Parsed devicetree nodes
    :param secure: Configure for a Trustzone secure system

    :return: Tuple of parsed of IPCT configuration records and IPCMAP records for use with UICR.
    """
    ipct_configs: List[IpctConfig] = []

    for node in nodes:
        channels = NodeChannels(
            owned=node.properties.get_val("owned-channels", []),
            child=node.properties.get_val("child-owned-channels", []),
            nonsecure=node.properties.get_val("nonsecure-channels", []),
        )

        if secure and node.secure:
            ns_channels: List[int] = channels.nonsecure
        else:
            ns_channels = copy(channels.owned)

        config = IpctConfig(
            instance=node,
            owned_channels=channels.owned,
            nonsecure_channels=ns_channels,
        )

        ipct_configs.append(config)

        _append_log_config(ConfigTypeEnum.IPCT, (channels, config))

    return ipct_configs


def _mboxes_get(node: ParsedDTNode) -> Dict[str, ParsedDTNode]:
    """
    Extract mbox nodes from IPC node.

    :param node: IPC node

    :return: Dictionary of mbox nodes with channel names as keys.
    """
    channel_names: Tuple[str, str] = ("tx", "rx")
    mbox_nodes: Dict[str, ParsedDTNode] = {}

    mbox_names: List[str] = node.properties["mbox-names"].val
    mboxes: List[edtlib.ControllerAndData] = node.properties["mboxes"].val

    for ch_name in channel_names:
        try:
            idx = mbox_names.index(ch_name)
        except ValueError:
            raise RuntimeError(f"Missing mbox name '{ch_name}' in node {node.name}")

        if not idx < len(mboxes):
            raise RuntimeError(
                f"Missing element {idx} ('{ch_name}') in mboxes property in node {node.name}"
            )

        mbox_nodes[ch_name] = ParsedDTNode(mboxes[idx].controller)

    return mbox_nodes


def _mboxes_are_valid(mbox_nodes: Dict[str, ParsedDTNode]) -> bool:
    """
    Check that the mbox nodes are enabled. Also check that the tx mbox is remote and
    rx mbox is local.

    :param mbox_nodes: Dictionary of mbox nodes

    :return: True if mbox nodes are enabled and configured correctly, otherwise False.
    """
    for ch_name in mbox_nodes.keys():
        if not mbox_nodes[ch_name].enabled:
            return False

    tx_mbox_is_remote: bool = mbox_nodes["tx"].compatibles_match(".*remote$")
    rx_mbox_is_local: bool = mbox_nodes["rx"].compatibles_match(".*local$")

    if not (tx_mbox_is_remote and rx_mbox_is_local):
        return False

    return True


def _remote_owner_id_get(mbox_nodes: Dict[str, ParsedDTNode]) -> OwnerID:
    """
    Get the owner ID associated with the remote mbox.

    :param mbox_nodes: Dictionary of mbox nodes

    :return: remote Owner ID.
    """

    processor_id = ProcessorID.from_nodelabels(mbox_nodes["tx"].labels)
    if processor_id is None:
        raise RuntimeError(
            f"Unable to determine processor for {mbox_nodes['tx'].path}, {mbox_nodes['tx'].labels}"
        )

    return OwnerID.from_processor(processor_id)


def _shmem_is_secure(shmem_node: ParsedDTNode) -> bool:
    """
    Check if a memory node is secure. That is, it has an ancestor
    with owned memory compatible and perm-secure property.
    It stops at the first node with compatible owned memory.

    :param shmem_node: Node to check security attribute of.

    :return: True if node is secure, otherwise False.
    """
    n: ParsedDTNode = shmem_node
    while n.path != "/":
        if n.compatibles_match(MEM_COMPAT_REGEX):
            return n.properties.get_val("perm-secure", False)
        n = ParsedDTNode(n.node.parent)
    return False


def _create_mailbox_config(
    shm_node: ParsedDTNode, remote_owner_id: OwnerID
) -> MailboxConfig:
    """
    Create MAILBOX record for a shared memory region.

    :param index: Mailbox record index
    :param shm_node: Shared memory to record in the MAILBOX UICR
    :param remote_owner_id: Owner on the remote side of the shared memory

    :return: Records for MAILBOX
    """
    return MailboxConfig(
        address=shm_node.address,
        size=shm_node.length,
        owner=remote_owner_id,
        secure=_shmem_is_secure(shm_node),
    )


def extract_mailbox_configs(
    nodes: List[ParsedDTNode],
) -> List[Union[DualMailboxConfig, MailboxConfig]]:
    """
    Extract MAILBOX record information from a list of parsed devicetree nodes.

    NB! Mailbox ordering matters for TX/RX relationships and should not be sorted
    from what is constructed here.

    :param nodes: Parsed devicetree nodes

    :return: List of MAILBOX configuration records for use with UICR.
    """
    configs: List[Union[DualMailboxConfig, MailboxConfig]] = []

    for node in nodes:
        mbox_nodes: Dict[str, ParsedDTNode] = _mboxes_get(node)
        if not _mboxes_are_valid(mbox_nodes):
            continue

        remote_owner: OwnerID = _remote_owner_id_get(mbox_nodes)
        if remote_owner != OwnerID.SECURE:
            continue

        if ResourceCompatible.MAILBOX_ICMSG.value in node.compatibles:
            tx_shm_node: ParsedDTNode = ParsedDTNode(node.properties["tx-region"].val)
            rx_shm_node: ParsedDTNode = ParsedDTNode(node.properties["rx-region"].val)
            config = DualMailboxConfig(
                tx=_create_mailbox_config(tx_shm_node, remote_owner),
                rx=_create_mailbox_config(rx_shm_node, remote_owner),
            )
        elif ResourceCompatible.MAILBOX_RPMSG.value in node.compatibles:
            shm_node: ParsedDTNode = ParsedDTNode(node.properties["memory-region"].val)
            config = _create_mailbox_config(shm_node, remote_owner)

        configs.append(config)

        _append_log_config(ConfigTypeEnum.MAILBOX, config)

    return configs


def extract_ptrextuicr_config(uicr_ext_node: ParsedDTNode) -> PtrExtUicrConfig:
    """
    Extract a PTREXTUICR record from devicetree

    :param uicr_ext_node: Extended UICR DT node
    :return: Record for PTREXTUICR
    """

    if (address := uicr_ext_node.address) is None:
        raise RuntimeError(
            "Unable to extract address of UICR extended region. "
            f"{uicr_ext_node} is missing a 'reg' property."
        )

    config = PtrExtUicrConfig(address=address)

    _append_log_config(ConfigTypeEnum.PTREXTUICR, config)

    return config


def _append_log_config(config_type: ConfigTypeEnum, info: Any):
    """
    Wrapper function for appending logging information to the global tracker and centralizing the
    implementation of updating it in a single place.

    :param config_type: Logging config group.
    :param info: Function-defined information for logging.
    """

    UICR_LOGS[config_type].append(info)


def _log_vtor_configs(configs: List[Union[SecureVtorConfig, NonSecureVtorConfig]]):
    """Log VTOR configurations extracted from UICR."""

    for config in configs:
        secure = isinstance(config, SecureVtorConfig)
        log_vrb(
            f"{'' if secure else 'Non'}Secure Image VTOR: 0x{config.initial_vtor:08x}"
        )
        log_vrb("")


def _log_memory_configs(log_info: List[Tuple[ParsedDTNode, MemoryConfig]]):
    """Log memory configurations extracted from UICR."""

    for i, info in enumerate(log_info):
        nodes, config = info

        log_vrb(f"Memory Config {i}:")

        labels = ", ".join(n.labels[0] if n.labels else n.name for n in nodes)
        log_vrb(f"\tAddress: 0x{config.address:08x} ({labels})")
        log_vrb(
            f"\tSize: {hex(config.size)} (Ends @0x{config.address + config.size:08x})"
        )
        log_vrb(f"\tOwner: {config.owner.id_string()}")
        if config.non_secure_callable:
            log_vrb(f"\tIs non-secure callable (no permissions)")
        else:
            log_vrb(
                f"\tPermissions: {'R' if config.readable else ''}{'W' if config.writable else ''}{'X' if config.executable else ''}"
            )
            log_vrb(f"\tSecure: {bool(config.secure)}")
            log_vrb(f"\tSecure DMA: {bool(config.secure)}")
        log_vrb("")


def _log_periph_configs(log_info: List[Tuple[bool, PeripheralConfig]]):
    """Log peripheral configurations extracted from UICR."""

    for i, info in enumerate(log_info):
        reserved = info[0]
        config = info[1]

        log_vrb(f"Peripheral Config {i}:")
        log_vrb(f"\tAddress: {config.instance.label_string()}")
        log_vrb(f"\tSecure: {config.secure}")
        log_vrb(f"\tSecure DMA: {config.secure}")
        log_vrb(f"\tIRQ routed to CPU: {config.processor.id_string()}")
        if reserved:
            log_vrb(f"\tReserved for domain subprocessor")
        log_vrb("")


def _log_gpiote_configs(log_info: List[Tuple[NodeChannels, GpioteConfig]]):
    """Log GPIOTE configurations extracted from UICR."""

    for i, info in enumerate(log_info):
        channels = info[0]
        config = info[1]

        log_vrb(f"GPIOTE Config {i}:")
        log_vrb(f"\tInstance: {config.instance.label_string()}")
        log_vrb(f"\t{channels}")
        log_vrb("")


def _log_grtc_configs(log_info: List[NodeChannels]):
    """Log GRTC configurations extracted from UICR."""

    for i, channels in enumerate(log_info):
        log_vrb(f"GRTC Config {i}:")
        log_vrb(f"\t{channels}")
        log_vrb("")


def _log_dppi_configs(log_info: List[DppiLogInfo]):
    """Log DPPIC configurations extracted from UICR."""

    for i, info in enumerate(log_info):
        if info.mapped_indexing:
            log_dbg(f"- Using mapped instance indexing for UICR.DPPIC[{i}]")

        if info.separate_source_sink:
            log_dbg(
                f"- Using separate SOURCE/SINK registers for UICR.DPPIC[{i}] channels"
            )

        log_vrb("")

        log_vrb(f"DPPI Config {i}:")
        log_vrb(f"\tInstance: {info.config.instance.label_string()}")
        log_vrb(f"\t{info.channels}")
        log_vrb(f"\tChannel Groups: {else_none(info.config.owned_channel_groups)}")
        log_vrb("")


def _log_mailbox_configs(configs: List[Union[DualMailboxConfig, MailboxConfig]]):
    """Log MAILBOX configurations extracted from UICR."""

    log_entry_idx = 0

    def print_mailbox(idx: int, mbx: MailboxConfig, rx_tx=None, rx_tx_id=None):
        log_vrb(f"Mailbox Config {idx}:")
        log_vrb(f"\tAddress: 0x{mbx.address:08x}")
        log_vrb(f"\tSize: {hex(mbx.size)} (Ends @0x{mbx.address + mbx.size:08x})")
        log_vrb(f"\tOwner: {mbx.owner.id_string()}")
        log_vrb(f"\tSecure: {bool(mbx.secure)}")
        if rx_tx and rx_tx_id is not None:
            log_vrb(f"\tIs {rx_tx} to MAILBOX[{rx_tx_id}]")
        log_vrb("")

    for config in configs:
        if isinstance(config, DualMailboxConfig):
            tx_idx = log_entry_idx
            rx_idx = log_entry_idx + 1
            print_mailbox(tx_idx, config.tx, rx_tx="TX", rx_tx_id=rx_idx)
            print_mailbox(rx_idx, config.rx, rx_tx="RX", rx_tx_id=tx_idx)
            log_entry_idx += 2
        else:
            print_mailbox(log_entry_idx, config)
            log_entry_idx += 1


def _log_ipct_configs(ipct_log_info, ipcmap_log_info):
    """Log IPCT configurations extracted from UICR."""

    global_ipcmaps = [
        x
        for x in ipcmap_log_info
        if DomainID.from_address(x.node.address) == DomainID.GLOBAL
    ]
    local_ipcmaps = [x for x in ipcmap_log_info if x not in global_ipcmaps]

    def log_source_sinks(log_info):
        log_vrb("\tChannel Mapping:")
        for map_ in log_info.source_map:
            sink_target = (
                f"{map_.sink_domain.name}, "
                if map_.sink_domain != map_.source_domain
                else ""
            )
            log_vrb(
                f"\t\tCH{map_.source_channel} -> {sink_target}CH{map_.sink_channel}"
            )
        for map_ in log_info.sink_map:
            source_target = (
                f"{map_.source_domain.name}, "
                if map_.sink_domain != map_.source_domain
                else ""
            )
            log_vrb(
                f"\t\tCH{map_.sink_channel} <- {source_target}CH{map_.source_channel}"
            )

    log_configs = 0

    for i, info in enumerate(ipct_log_info):
        log_configs = i
        channels = info[0]
        config = info[1]

        instance_ipcmap = [
            x for x in global_ipcmaps if x.node.address == config.instance.address
        ][0]

        log_vrb(f"IPCT Config {i}")
        log_vrb(f"\tInstance: {config.instance.label_string()}")
        log_vrb(f"\t{channels}")
        log_source_sinks(instance_ipcmap)
        log_vrb("")

    for ipcmap in local_ipcmaps:
        log_configs += 1
        log_vrb(f"IPCT Config {log_configs}:")
        log_vrb(f"\tInstance: {ipcmap.node.label_string()}")
        log_source_sinks(ipcmap)
        log_vrb("")


def _log_gpio_configs(gpio_log_info):
    """Log GPIO configurations extracted from UICR."""

    pinctrl_nodes = defaultdict(list)

    for info in gpio_log_info:
        port = info.gpio.port
        pin = info.gpio.pin
        ctrlsel = info.gpio.ctrlsel
        secure = info.gpio.secure

        if isinstance(info, PinctrlLogInfo):
            # Group them by nodes to avoid duplicate pin outputs, since pinctrl can have multiple
            # nodelabel alternatives reference the same pin info.
            pinctrl_nodes[f"P{port}.{pin}"].append(info)

        elif isinstance(info, GpiosLogInfo):
            log_vrb(f"P{port}.{pin}")
            if ctrlsel is not None:
                log_vrb(f"\tCTRLSEL: {ctrlsel}")
            log_vrb(f"\tSecure: {bool(secure)}")
            log_vrb(f"\tNode: {info.ctrldata.node.name}")
            log_dbg(f"\tSource: {info.ctrldata.node.path}")

            if info.ctrldata.controller.labels:
                log_dbg(
                    f"\tgpios: <&{info.ctrldata.controller.labels[0]} {pin} {info.cell_data['data']['flags']}>"
                )
            else:
                log_dbg(
                    f"\tController: {info.ctrldata.controller.name}@{info.ctrldata.controller.path}"
                )
                log_dbg(f"\tProperties: <{info.cell_data}>")

        else:
            log_vrb(f"P{port}.{pin}")
            if ctrlsel != CTRLSEL_DEFAULT:
                log_vrb(f"\tCTRLSEL: {ctrlsel}")
            log_vrb(f"\tSecure: {bool(secure)}")
            node_name = (
                f"{info.prop.node.labels[0]}"
                if info.prop.node.labels
                else f"{info.prop.node.name}"
            )
            log_vrb(f"\tSource: {node_name}@{info.prop.node.path}")
            log_dbg(f"Property: {info.prop}")

    for pinset, nodes in pinctrl_nodes.items():
        log_vrb(pinset)
        if nodes[-1].gpio.ctrlsel is not None:
            log_vrb(f"\tCTRLSEL: {nodes[-1].gpio.ctrlsel}")
        log_vrb(f"\tSecure: {bool(secure)}")
        log_vrb(f"\tPinctrl:")
        for node in nodes:
            log_vrb(f"\t\tNode: {node.config_node.name}")
            log_dbg(f"\t\tSource: {node.group_node.path}")
            log_dbg(f"\t\tProperty: {node.group_node.props['psels']}")


def _log_extuicr_configs(ptrextuicr_log_info, gpio_log_info):
    """
    Log all relevant configuration types that are part of EXTUICR from their extracted devicetree
    information.
    """

    ptrextuicr_config = ptrextuicr_log_info[0]

    log_vrb(f"Location: 0x{ptrextuicr_config.address:08x}")
    log_vrb("")
    if gpio_log_info:
        log_vrb("--------- GPIO ---------")
        log_vrb("")
        _log_gpio_configs(gpio_log_info)


def _log_extracted_uicr_configs(uicr):
    """
    Dump all UICR/EXTUICR configurations extracted from devicetree that were appended to the global
    UICR logging map.

    This function and its internals allow greater flexibility in the order and output styling of
    user-friendly info dumps for UICR contents (such as associating IPCMAPs with their actual IPCTs
    in the IPCT logging section, when their UICR configurations are independent), as well as
    reducing the clutter of logging formats and calls in the associated extraction functions.
    """

    log_vrb(f"Location: 0x{uicr.address:08x}")
    log_vrb(str(uicr.processor))

    def print_section_header(name: str):
        """Internal helper for printing consistent headers."""
        log_vrb("")
        log_vrb(f"========= {name} =========")
        log_vrb("")

    if UICR_LOGS.get(ConfigTypeEnum.VTOR):
        print_section_header(ConfigTypeEnum.VTOR.name)
        _log_vtor_configs(UICR_LOGS.pop(ConfigTypeEnum.VTOR))

    if UICR_LOGS.get(ConfigTypeEnum.MEMORY):
        print_section_header(ConfigTypeEnum.MEMORY.name.capitalize())
        _log_memory_configs(UICR_LOGS.pop(ConfigTypeEnum.MEMORY))

    if UICR_LOGS.get(ConfigTypeEnum.PERIPH):
        print_section_header("Peripherals")
        _log_periph_configs(UICR_LOGS.pop(ConfigTypeEnum.PERIPH))

    if UICR_LOGS.get(ConfigTypeEnum.DPPI):
        print_section_header(ConfigTypeEnum.DPPI.name)
        _log_dppi_configs(UICR_LOGS.pop(ConfigTypeEnum.DPPI))

    if UICR_LOGS.get(ConfigTypeEnum.GRTC):
        print_section_header(ConfigTypeEnum.GRTC.name)
        _log_grtc_configs(UICR_LOGS.pop(ConfigTypeEnum.GRTC))

    if UICR_LOGS.get(ConfigTypeEnum.MAILBOX):
        print_section_header("Mailbox (IPC)")
        _log_mailbox_configs(UICR_LOGS.pop(ConfigTypeEnum.MAILBOX))

    if UICR_LOGS.get(ConfigTypeEnum.IPCT):
        print_section_header(ConfigTypeEnum.IPCT.name)
        _log_ipct_configs(
            UICR_LOGS.pop(ConfigTypeEnum.IPCT), UICR_LOGS.pop(ConfigTypeEnum.IPCMAP)
        )

    if UICR_LOGS.get(ConfigTypeEnum.GPIOTE):
        print_section_header(ConfigTypeEnum.GPIOTE.name)
        _log_gpiote_configs(UICR_LOGS.pop(ConfigTypeEnum.GPIOTE))

    if UICR_LOGS.get(ConfigTypeEnum.GPIO) and not UICR_LOGS.get(
        ConfigTypeEnum.PTREXTUICR
    ):
        raise RuntimeError(
            "Missing UICR.PTREXTUICR; required for UICR GPIO configurations"
        )

    if UICR_LOGS.get(ConfigTypeEnum.PTREXTUICR):
        print_section_header("Extended UICR")
        _log_extuicr_configs(
            UICR_LOGS.pop(ConfigTypeEnum.PTREXTUICR),
            UICR_LOGS.pop(ConfigTypeEnum.GPIO, []),
        )
