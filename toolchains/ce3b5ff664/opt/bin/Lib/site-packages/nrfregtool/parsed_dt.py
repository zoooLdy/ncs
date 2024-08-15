#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
This module provides helper functions and wrappers around edtlib devicetree objects that emphasize
aspects that are relevant in a product context, such as memory region and peripheral ownership.
"""

from __future__ import annotations

import collections.abc
import re
from abc import ABC, abstractmethod
from functools import cached_property, total_ordering
from itertools import groupby
from pprint import pformat
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Union

from devicetree import edtlib

from .common import AddressOffset, AddressRegion, DomainID, ProcessorID, Record


# Regex for known partition compatibles, which affect how the address of a
# parsed devicetree node should be interpreted.
PARTITION_COMPAT_REGEX = re.compile(r"^(fixed|nordic,owned)-partitions$")


class PropertyMap(collections.abc.Mapping):
    """Read-only map of devicetree properties"""

    def __init__(self, properties: Dict[str, edtlib.Property]):
        self._proxy = MappingProxyType(properties)

    def get_val(self, name: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get the value of a devicetree property defined in the property map.
        Returns a default value if the property does not exist.

        :param name: Property name
        :param default: Default value to return if the property does not exist

        :return: Property value
        """
        if name not in self._proxy:
            return default
        return self._proxy[name].val

    def __getitem__(self, name: str) -> edtlib.Property:
        return self._proxy.__getitem__(name)

    def __iter__(self) -> Iterator[str]:
        return self._proxy.__iter__()

    def __len__(self) -> int:
        return self._proxy.__len__()

    def __str__(self) -> str:
        return f"<PropertyMap 0x{hex(id(self))}: {pformat(dict(self))}>"


@total_ordering
class ParsedDTNode:
    def __init__(self, node: edtlib.Node):
        """Initialize a ParsedDTNode from a devicetree node."""
        self._node: edtlib.Node = node

    @cached_property
    def name(self) -> str:
        """Name of the node"""
        return self.node.name.split("@")[0]

    @property
    def node(self) -> edtlib.Node:
        """The underlying edtlib.Node wrapped by this instance"""
        return self._node

    @property
    def status(self) -> str:
        """Status of the node."""
        return self.node.status

    @property
    def enabled(self) -> bool:
        """Node has status='okay'"""
        return self.status == "okay"

    @property
    def reserved(self) -> bool:
        """Node has status='reserved'"""
        return self.status == "reserved"

    @cached_property
    def parent(self) -> Optional[ParsedDTNode]:
        """Parent node"""
        if not self.node.parent:
            return None
        return ParsedDTNode(self.node.parent)

    @cached_property
    def address(self) -> Optional[int]:
        """Absolute address of the node"""
        if not self.node.regs:
            return None

        addr = self.node.regs[0].addr
        parent = self.parent

        if parent and parent.compatibles_match(PARTITION_COMPAT_REGEX):
            gparent = parent.node.parent
            if gparent:
                if not gparent.regs:
                    raise RuntimeError(f"Missing 'reg' property in node {gparent.path}")
                addr += gparent.regs[0].addr

        return addr

    @property
    def length(self) -> Optional[int]:
        """Address space size of the node"""
        if not self.node.regs:
            return None
        return self.node.regs[0].size

    @property
    def path(self) -> str:
        """Devicetree node path of the node"""
        return self.node.path

    @cached_property
    def secure(self) -> Optional[bool]:
        """
        Security attribute of the node, determined solely by its bus. In Trustzone-enabled systems,
        this is determined by the security bit in the node's absolute address.

        This attribute does not represent any other context of security that could relate to the
        node, such as properties denoting secure permissions, memory, etc.
        """
        if self.address is None:
            return None

        address = self.node.bus_node.unit_addr if self.node.bus_node else self.address

        return bool(address & (1 << AddressOffset.SECURITY))

    @cached_property
    def properties(self) -> PropertyMap:
        """Devicetree properties defined on the node"""
        return PropertyMap(self.node.props)

    @property
    def labels(self) -> List[str]:
        """Labels"""
        return self.node.labels

    @property
    def compatibles(self) -> List[str]:
        """Compatible property strings if one or more exist, else empty."""
        return self.node.compats

    @property
    def domain(self) -> Optional[DomainID]:
        """Domain ID associated to the node's bus address."""
        if self.address is None:
            return None
        return DomainID.from_address(self.address)

    @property
    def region(self) -> Optional[AddressRegion]:
        """Address region associated to the node's bus address."""
        if self.address is None:
            return None
        return AddressRegion.from_address(self.address)

    @property
    def pinctrls(self) -> List[edtlib.PinCtrl]:
        """Pinctrl configurations referenced by the node"""
        return self.node.pinctrls

    def __str__(self) -> str:
        """String representation of the class."""

        attrs = {
            "Properties": [str(v) for v in self.properties.values()],
            "Nodelabels": self.labels,
            "Path": self.path,
        }
        address_str = "" if self.address is None else f"@{hex(self.address)}"
        domain_str = (
            ""
            if self.domain is None
            else f" - {self.domain.name} Domain ({self.domain})"
        )
        return (
            f"ParsedDTNode {self.name}{address_str}{domain_str}: " f" {pformat(attrs)}"
        )

    def label_string(self, upper: bool = True) -> str:
        """
        Helper for returning an instance string of the node.

        :return: Formatted instance string.
        """
        label = (
            f" ({self.labels[0].upper() if upper else self.labels[0]})"
            if self.labels
            else ""
        )
        return f"0x{self.address:08x}{label}"

    def compatibles_contain(self, substring: str) -> bool:
        """
        Helper method for checking the existence of a substring in all of the node's compatible
        strings.

        :param substring: Compatible substring to check against

        :return: True if any compatible contains the substring, False otherwise.
        """

        return any(substring in compatible for compatible in self.compatibles)

    def compatibles_match(self, pattern: Union[str, re.Pattern]) -> bool:
        """
        Helper method for checking if a regular expression matches any of the node's compatible
        strings.

        :param pattern: Pattern to check against

        :return: True if any compatible matches the pattern, False otherwise.
        """

        return any(
            re.match(pattern, compatible) is not None for compatible in self.compatibles
        )

    def __eq__(self, other: object) -> bool:
        """Equality operator based on the node's dep_ordinal value"""

        if not isinstance(other, ParsedDTNode):
            return False
        # edtlib.Node.dep_ordinal uniquely identifies a node in a given EDT
        return (
            self._node.edt is other._node.edt
            and self._node.dep_ordinal == other._node.dep_ordinal
        )

    def __lt__(self, other: object) -> bool:
        """Less than operator based on the node's address, size and name (in that order)"""

        if not isinstance(other, ParsedDTNode):
            return NotImplemented

        return (self.address, self.length, self.name) < (
            other.address,
            other.length,
            other.name,
        )

    def __hash__(self) -> int:
        """Hash operator based on the node's dep_ordinal value"""

        return hash((id(self._node.edt), self._node.dep_ordinal))


class InterpretedPeripheral(ABC):
    """
    Abstract base class of a peripheral that is interpreted from devicetree configurations.
    """

    def __init__(self, devicetree: edtlib.EDT, name: str):
        """
        Initialize the class attribute(s).

        :param devicetree: edtlib Devicetree object
        :param name: Peripheral name
        """
        self._name = name.upper()
        self._properties: Dict[str, edtlib.Property] = {}
        self._chosen_nodes: Dict[str, ParsedDTNode] = {
            name: ParsedDTNode(node) for name, node in devicetree.chosen_nodes.items()
        }

    @property
    def name(self) -> str:
        """
        Name of the interpreted peripheral.
        """
        return self._name

    @property
    @abstractmethod
    def address(self) -> int:
        """
        Base address of the peripheral.
        """
        ...

    @cached_property
    def properties(self) -> PropertyMap:
        """
        Peripheral properties defined for the peripheral in devicetree.
        """
        return PropertyMap(self._properties)

    @property
    def chosen(self) -> Dict[str, ParsedDTNode]:
        """
        Chosen nodes in the devicetree.
        """
        return self._chosen_nodes

    def __str__(self) -> str:
        """String representation of the class."""

        props = {
            "Properties": [str(v) for v in self.properties.values()],
            "Chosen": self.chosen,
        }
        nodes = {k.name: [str(n) for n in v] for k, v in self._resource_nodes.items()}
        return f"Parsed {self.name}:\n{pformat(props)}\n{pformat(nodes)}"


class ProcessorInfo(NamedTuple):
    """
    Helper tuple for containing relevant information about the parsed CPU.
    """

    cpu_type: str
    cpu_id: ProcessorID

    def __repr__(self) -> str:
        """
        Basic class representation
        """
        return f"<ProcessorInfo at {hex(id(self))} cpu_type={self.cpu_type}, cpu_id={self.cpu_id}>"

    def __str__(self) -> str:
        """
        String representation of the processor info.
        """
        make, model = (x.capitalize() for x in self.cpu_type.split(",", maxsplit=1))
        return f"Processor {self.cpu_id} - {make} {model}"


class PeripheralResult(NamedTuple):
    """
    Information about a peripheral that is extracted from a devicetree.
    """

    base_address: int
    record: Record


def dt_processor_info(devicetree: edtlib.EDT) -> ProcessorInfo:
    """
    Get processor information from a domain's devicetree.

    :param devicetree: Devicetree

    :return: Tuple of processor information of the devicetree's domain
    """
    cpus = [
        node
        for node in devicetree.get_node("/cpus").children.values()
        if node.name.startswith("cpu@")
    ]
    if len(cpus) != 1:
        raise RuntimeError(
            f"Expected exactly 1 'cpu' node, but devicetree contained {len(cpus)} nodes"
        )

    cpu = cpus[0]
    try:
        compatible = cpu.compats[0]
    except IndexError:
        raise RuntimeError("Devicetree 'cpu' node has no compatible")

    processor_id = ProcessorID.from_value(cpu.regs[0].addr)
    if processor_id is None:
        raise RuntimeError(
            f"Devicetree 'cpu' node has invalid Processor ID {cpu.regs[0].addr}"
        )

    return ProcessorInfo(compatible, processor_id)


class NodeChannels:
    """Helper class for encapsulating channel resources for a node under a common object."""

    __slots__ = ["_owned", "_child", "_nonsecure", "_all", "_source", "_sink"]

    def __init__(
        self,
        owned: List[int] = [],
        child: List[int] = [],
        nonsecure: List[int] = [],
        source: List[int] = [],
        sink: List[int] = [],
    ):
        self._owned = owned
        self._child = child
        self._nonsecure = nonsecure
        self._all = list(set(self._owned) | set(self._child))
        self._source = source
        self._sink = sink

    @property
    def owned(self) -> List[int]:
        """Privately owned channels by the node."""
        return self._owned

    @property
    def child(self) -> List[int]:
        """Channels owned by the node on behalf of a child subprocessor."""
        return self._child

    @property
    def nonsecure(self) -> List[int]:
        """Channels owned by the node that are nonsecure."""
        return self._nonsecure

    @property
    def all(self) -> List[int]:
        """All channels owned by the node."""
        return self._all

    @property
    def source(self) -> List[int]:
        """Channels owned by the node that are configured as sources."""
        return self._source

    @property
    def sink(self) -> List[int]:
        """Channels owned by the node that are configured as sinks."""
        return self._sink

    def __str__(self) -> str:
        """String representation of the class."""
        if self.child or self.nonsecure or self.source or self.sink:
            channels = {
                f"{'Self-o' if self.child else 'O'}wned": self.owned,
                "Child-owned": self.child,
                "Nonsecure": self.nonsecure,
                "Sources": self.source,
                "Sinks": self.sink,
            }
            info_str = "\n" + "\n".join(
                [f"\t\t{k}: {v}" for k, v in channels.items() if v]
            )
        else:
            info_str = f" {self.owned}"

        return f"Channels:{info_str}"


class IndexedByInstance(NamedTuple):
    """Index by instance address"""

    # Instance node to index the configuration by
    instance: ParsedDTNode


class IndexedByPosition(NamedTuple):
    """Index by array position"""

    # Number of array slots taken up by one configuration object
    num_slots: int


# Methods for indexing configuration objects
IndexedBy = Union[IndexedByInstance, IndexedByPosition]


class ExtractedConfig(ABC):
    """Base class for extracted configuration objects."""

    @property
    def indexed_by(self) -> Optional[IndexedBy]:
        """
        If the configuration is meant for an array register, return metadata describing how to
        assign an array index to the  configuration.
        If None, the configuration is not meant for an array register and does not require an index.
        """
        return None

    @abstractmethod
    def to_record(self, *, index: int, **kwargs) -> Record:
        """
        Convert object to a register Record representation.
        The return value contains both the Record and the number of register slots taken up by the
        Record. The number of slots can be used to assign the Record to the correct register index.

        :param index: Index of the register to assign the Record to.
        :param kwargs: Unspecified keyword arguments to use in constructing the Record.
        :return: A tuple of the Record and the number of registers represented in the Record.
        """
        raise NotImplementedError


def sort_configs_by_type(configs: List[ExtractedConfig]) -> List[List[ExtractedConfig]]:
    """
    Sort configs by type, then group by type and sort each group internally.
    This ensures that the order of the configs in the output record is consistent.
    """
    type_sorted_configs = sorted(configs, key=lambda c: type(c).__name__)

    return [sorted(c) for _, c in groupby(type_sorted_configs, key=type) if c]
