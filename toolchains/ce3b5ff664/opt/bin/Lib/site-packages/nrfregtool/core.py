#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import importlib.resources
from collections import ChainMap, Counter
from functools import cached_property
from itertools import chain
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import svd
import tomli
from devicetree import edtlib
from intelhex import IntelHex
from more_itertools import more
from svd import Device, EPath, Field, Peripheral, Register

from . import bicr, uicr
from .common import Record, log_dbg, log_vrb, log_dev
from .parsed_dt import PeripheralResult

# TOML format constants
TOML_FIELD_AS_HEX_WHEN_LARGER_THAN = 32
TOML_FIELD_COMMENT_START_AT_COLUMN = 30
TOML_FIELD_DEFAULT_VALUE_START_AT = 56


# Peripherals that have DTS parsing/interpreting implementations
DTS_PARSEABLE_PERIPHERALS = ["UICR", "BICR"]

# Registers from supported peripherals that must be set as unconstrained
# regardless of commandline options.
REQUIRED_UNCONSTRAINED = {
    "BICR": bicr.BICR_UNCONSTRAINED,
}

# Preset options for removing registers from the SVD file.
# Removing these registers is a hack to deal with outdated SVD files.
SKIP_REGISTERS = {
    "UICR": {
        # These registers were moved to UICREXTENDED or are removed outright, but remain in UICR
        # in some SVD files.
        r"UICR(?!EXTENDED).*": ["GPIO[%s]", "IPCT.LOCAL"]
    }
}


# A map of values for a given field that are conflicting between different versions of the device
CONFLICTING_FIELD_VALUES = {
    "BICR": {
        "MODE": {
            # 'Pierce' and 'Crystal' are both used for '0' for this field
            "Pierce": 0,
            "Crystal": 0,
        }
    }
}

# Peripherals that reside in NVM
NVM_PERIPHERALS = ["UICR", "BICR"]
NVM_FILL_BYTE = 0xFF
NVM_RESET_VALUE = 0xFFFF_FFFF

# Split constants
MIN_UICR_DIFF_DELTA = 150  # Arbitrary number that produced nice UICR results


class LogicalPeripheral:
    """Abstraction for a set of peripherals that are logically grouped together."""

    def __init__(
        self,
        name: str,
        device: Device,
        record: Record,
        unconstrained: Optional[List[str]] = None,
        base_address: Optional[int] = None,
        mem_map: Optional[MemoryMap] = None,
    ):
        """
        :param name: Name of the peripheral.
        :param device: SVD device element
        :param record: Peripheral register record containing register values
        :param unconstrained: List of register paths that should not be constrained
        :param base_address: Base address of the peripheral
        :param mem_map: The memory map containing register values
        """

        self._name: str = name
        self._device: Device = device

        main_peripheral = _find_peripheral(self._device, self._name)
        if base_address is not None:
            main_peripheral = main_peripheral.copy_to(base_address)

        self._peripherals: List[Peripheral] = [main_peripheral]

        if name == "UICR":
            uicrext_address = 0xFFFF_FFFF

            if mem_map is not None:
                ptrextuicr_address = main_peripheral["PTREXTUICR"].address
                uicrext_address = mem_map.get(ptrextuicr_address, uicrext_address)

            if (ptrextuicr := record.get("ptrextuicr_0")) is not None:
                uicrext_address = ptrextuicr.get("PTREXTUICR", uicrext_address)

            # Only generate extended UICR if it has a non-default address
            if uicrext_address != 0xFFFF_FFFF:
                try:
                    uicrext_dev = _find_peripheral(self._device, "UICREXTENDED")
                except LookupError:
                    log_dev("Default SVD has no extended UICR")
                    uicrext_dev = self._load_uicrextended()

                # This hack ensures that UICREXTENDED reset values are properly aligned with UICR
                # and NVM regions. If they are defined as 0 from the MDK, or inherited as 0 from
                # its parent device if not defined at all, the extended UICR region will be
                # zeroed out and appear as if everything is to be allocated.
                if uicrext_dev._reg_props.reset_value == 0:
                    uicrext_dev._reg_props.reset_value = NVM_RESET_VALUE
                    log_dev(
                        f"Aligned {uicrext_dev.name} reset value with NVM: {uicrext_dev._reg_props.reset_value}"
                    )

                self._peripherals.append(uicrext_dev.copy_to(uicrext_address))

        if unconstrained is not None:
            self._unconstrain(unconstrained)

        if mem_map is not None:
            self._fill_from_mem_map(mem_map)

        self._fill_from_record(record)

    @staticmethod
    def _load_uicrextended() -> Peripheral:
        """Load the UICREXTENDED peripheral from the separate, bundled SVD file."""

        _path = ["nrfregtool.resources", "uicrextended.svd"]

        with importlib.resources.path(*_path) as uicrext_svd_path:
            bundled_uicrext = svd.parse(
                uicrext_svd_path,
                options=svd.Options(
                    ignore_overlapping_structures=True,
                    parent_relative_cluster_address=True,
                ),
            )

        log_dev(f"Extended UICR extracted from: {Path(*_path)}")

        return _find_peripheral(bundled_uicrext, "UICREXTENDED")

    def _unconstrain(self, unconstrained: List[str]):
        """
        Remove restrictions put on a value held by the register.

        :param unconstrained: List of register paths that should not be constrained
        :raises ValueError: If a register is not found in the peripheral set
        """
        for register_path in unconstrained:
            record_name, field_name = split_register_path(register_path)
            try:
                register = self.record_name_to_register[record_name]
                if field_name is None:
                    register.unconstrain()
                else:
                    register[field_name].unconstrain()
            except KeyError as e:
                raise ValueError(
                    f"Unconstrained register path {register_path} not found in peripheral set "
                    f"{[p.name for p in self._peripherals]}"
                ) from e

    def _fill_from_record(self, record: Record):
        """
        Fill the values of the peripheral set based on the contents of an instance of Record.

        :param record: The record that register values are filled from
        :raises ValueError: If the register cannot be found in the peripheral set
        """
        for record_name, record_fields in record.items():
            try:
                register = self.record_name_to_register[record_name]
            except KeyError as e:
                if self._name == "UICR" and "_instance" in record_name:
                    log_dbg(f"Skipping register {record_name}")
                    continue
                else:
                    raise ValueError(
                        f"Register with record name '{record_name}' not found in peripheral set "
                        f"{[p.name for p in self._peripherals]}"
                    ) from e

            for field, value in record_fields.items():
                try:
                    register[field] = value
                except ValueError as e:
                    # The value is not found in the SVD. This could be the result of a register
                    # value enumeration having different names in different versions of the
                    # product. Check if the field in question is one of those fields, and load
                    # the raw value associated with that enumeration.
                    if (
                        self._name in CONFLICTING_FIELD_VALUES
                        and field in CONFLICTING_FIELD_VALUES[self._name]
                        and value in CONFLICTING_FIELD_VALUES[self._name][field]
                    ):
                        new_value = CONFLICTING_FIELD_VALUES[self._name][field][value]
                        log_dbg(
                            f"Setting {record_name}.{field} to {new_value} as the value '{value}'"
                            f" it is an aliased enumeration: {CONFLICTING_FIELD_VALUES}"
                        )
                        register[field] = new_value
                    else:
                        raise e

    def _fill_from_mem_map(self, mem_map: MemoryMap):
        """
        Fill the values of the peripheral set based on the contents of an instance of MemoryMap.

        :param mem_map: The memory map that register values are filled from
        :raises ValueError: If the register cannot be found in the peripheral set
        """
        for addr, value in mem_map.items():
            try:
                self.address_to_register[addr].content = value
            except KeyError:
                raise ValueError(
                    f"Address {hex(addr)} not found in peripheral set "
                    f"{[p.name for p in self._peripherals]}"
                )

    @property
    def name(self) -> str:
        """Name of the peripheral set."""
        return self._name

    @cached_property
    def address_to_register(self) -> Mapping[int, Register]:
        """Map of all registers in the peripheral set, keyed by address."""
        return ChainMap(
            *(
                {
                    register.address: register
                    for register in peripheral.register_iter(leaf_only=True)
                }
                for peripheral in self._peripherals
            )
        )

    @cached_property
    def record_name_to_register(self) -> Mapping[str, Register]:
        """Map from record name to register instance."""
        return ChainMap(
            *(
                _make_record_to_register_map(peripheral)
                for peripheral in self._peripherals
            )
        )

    @cached_property
    def path_to_record_name(self) -> Mapping[EPath, str]:
        """Map of register path to record name."""
        return {
            register.path: record_name
            for record_name, register in self.record_name_to_register.items()
        }

    @cached_property
    def memory_map(self) -> Mapping[int, int]:
        """
        Return a combined memory map of the peripherals in the set.
        The returned map has a granularity of 1 byte.
        """
        return ChainMap(
            *(dict(p.memory_iter(absolute_addresses=True)) for p in self._peripherals)
        )

    @cached_property
    def written_memory_map(self) -> Mapping[int, int]:
        """
        Return a combined memory map of the peripherals in the set, but only containing those
        entries that were explicitly written to.
        The returned map has a granularity of 1 byte.
        """
        return ChainMap(
            *(
                dict(p.memory_iter(absolute_addresses=True, written_only=True))
                for p in self._peripherals
            )
        )

    @property
    def address_ranges(self) -> List[range]:
        """Return a list of address ranges covered by each peripheral in the peripheral set."""
        return [range(*peripheral.address_bounds) for peripheral in self._peripherals]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self._peripherals}"

    def __str__(self):
        return self.to_str(lambda reg: reg.modified)

    def to_str(self, reg_filter: Optional[Callable[[Register], bool]] = None):
        lines = []

        lines.append("{")

        for peripheral in self._peripherals:
            lines.append(f"  {peripheral.name} @ 0x{peripheral.base_address:08x}")

            for reg in peripheral.register_iter(leaf_only=True):
                if reg_filter is None or reg_filter(reg):
                    lines.append(f"    {reg}")

        lines.append("}")

        return "\n".join(lines)


def split_register_path(register_path: str) -> Tuple[str, Optional[str]]:
    """Split a register path such as 'register_name.field_name' into its components"""
    path_components = register_path.strip(".").rsplit(".", maxsplit=1)
    register_name = path_components[0]
    field_name = path_components[1] if len(path_components) > 1 else None
    return register_name, field_name


def _make_record_to_register_map(peripheral: Peripheral) -> Dict[str, Register]:
    """
    Make a map from record name to register based on the registers in the given peripheral.
    It is assumed that each register name maps to a unique record name in the peripheral.
    """

    register_counts: Counter = Counter()
    record_to_register_map = {}

    for register in peripheral.register_iter(leaf_only=True):
        non_indexed_path = register.path.to_flat()
        index = register_counts[non_indexed_path]
        register_counts[non_indexed_path] += 1
        record_name = f"{'_'.join(non_indexed_path).lower()}_{index}"
        record_to_register_map[record_name] = register

    return record_to_register_map


def _find_peripheral(device: Device, name: str) -> Peripheral:
    """
    Find a peripheral with the given name in the device.
    Certain common prefixes and suffixes used for peripheral names can be omitted from the name.
    """

    matches = [
        periph
        for n, periph in device.items()
        if name in (n, _strip_prefixes_suffixes(n, ["GLOBAL_"], ["_NS", "_S"]))
    ]

    if not matches:
        raise LookupError(
            f"No peripheral with name containing '{name}' found in the SVD file"
        )

    elif len(matches) > 1:
        raise ValueError(
            f"More than one peripheral with name containing '{name}' found in the "
            f"SVD file: {matches}"
        )

    return matches[0]


def _strip_prefixes_suffixes(
    word: str, prefixes: List[str], suffixes: List[str]
) -> str:
    """
    Remove a prefix and suffix from the given string.
    Up to one prefix and/or suffix is removed - if multiple of the provided strings match then
    the first found is removed.

    :param word: String to strip prefixes and suffixes from.
    :param prefixes: List of prefixes to strip.
    :param suffixes: List of suffixes to strip.

    :return: String where prefixes and suffixes have been removed.
    """

    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix) :]
            break

    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]
            break

    return word


class MemoryMap(dict):
    """The representation of a memory."""

    def __init__(
        self,
        hex_files: Sequence[Path],
        alignment: int = 4,
        change_endianess: bool = True,
    ):
        """
        :param hex_files: An iterable of hex files paths used to fill the memory map.
        :param alignment: The address alignment. Defaults to 4.
        :param change_endianess: If the endianess has to be changed. Defaults to True.

        :raises ValueError: On alignment <= 0.
        """
        mem_map = {}

        if alignment <= 0:
            raise ValueError

        self._alignment = alignment

        for uicr_file in hex_files:
            ih_hex = IntelHex()
            ih_hex.loadhex(uicr_file)
            mem_map.update(ih_hex.todict())

        for i, (addr, value) in enumerate(mem_map.items()):
            addr_aligned = self._align(addr)

            if change_endianess:
                shift = (i % alignment) * 8
            else:
                shift = (alignment - (i % alignment)) * 8

            old_value = self.get(addr_aligned, 0)
            new_value = old_value | (value << shift)

            self.update({addr_aligned: new_value})

    def _align(self, addr: int):
        """Aligns the address.

        :param addr: The address to align.

        :return: An aligned address
        """
        return self._alignment * (addr // self._alignment)

    def __repr__(self):
        return "\n".join(
            [
                f"<0x{hex(addr)[2:].upper()} : 0x{hex(value)[2:].upper()}>"
                for addr, value in self.items()
            ]
        )


def parse_toml(*tomls: Sequence[Union[str, Path]]) -> Record:
    """
    Parse register records extracted from one or more TOML configurations.
    Later configurations override earlier ones.

    :param tomls: Sequence of TOML configuration files with records to load

    :return: Records of register content from TOML configuration files.
    """

    toml_record = Record()

    for config in tomls:
        with Path(config).open("rb") as f_toml:
            toml_record.update(tomli.load(f_toml))

    log_vrb("")
    log_vrb(f"========= TOML Configurations =========\n")
    log_vrb(pformat(toml_record.as_hex()))
    log_vrb("")

    return toml_record


def parse_dts(
    peripheral_name: str,
    dts: Union[str, Path],
    bindings_dirs: List[Path],
    use_pinctrl: bool = True,
) -> PeripheralResult:
    """
    Parse a compiled devicetree to extract records from its nodes.

    :param peripheral_name: Name of the peripheral to parse records for
    :param dts: Path to compiled devicetree file
    :param bindings_dirs: List of paths to directories containing devicetree bindings
    :param use_pinctrl: Use pinctrl for extracting GPIO records from peripheral nodes.

    :return: Records of register content applied by parsing the compiled devicetree file.
    """

    devicetree = edtlib.EDT(dts, bindings_dirs)

    if peripheral_name.upper() == "UICR":
        result = uicr.from_dts(devicetree, use_pinctrl=use_pinctrl)
    elif peripheral_name.upper() == "BICR":
        result = bicr.from_dts(devicetree)
    else:
        raise NotImplementedError(
            f"Devicetree parsing not supported for '{peripheral_name.upper()}'"
        )

    return result


def field_as_toml(field: Field, **kwargs) -> str:
    """
    TOML representation of a register bitfield.

    :kwarg comments: Append comment strings to the field assignment output
    :kwarg force_32_bit_fields: List of Fields that should be given as 32-bit integers, regardless
        of bit width (for example, addresses).

    :return: TOML string representation of the field contents.
    """
    comments = kwargs.get("comments")
    force_32_bit_fields = kwargs.get("force_32_bit_fields")

    reverse_enums: Dict[int, str] = {value: name for name, value in field.enums.items()}

    if field.content in reverse_enums:
        value = f'"{reverse_enums[field.content]}"'
        comment = ", ".join([enum for _value, enum in sorted(reverse_enums.items())])
        default = reverse_enums[field.reset_content]

    elif field.content > TOML_FIELD_AS_HEX_WHEN_LARGER_THAN:
        if isinstance(force_32_bit_fields, list) and any(
            x.upper() == field.name.upper() for x in force_32_bit_fields
        ):
            value = f"0x{field.content << field.bit_offset:08x}"
            if isinstance(field.allowed_values, range):
                comment = (
                    f"{field.allowed_values.start}.."
                    f"0x{(field.allowed_values.stop - 1) << field.bit_offset:08x}"
                )
            else:
                comment = ", ".join([f"{v} (0x{v:08x})" for v in field.allowed_values])
            default = f"0x{field.reset_content << field.bit_offset:08x}"
        else:
            value = hex(field.content)
            if isinstance(field.allowed_values, range):
                comment = f"0..0x{field.allowed_values.stop - 1:x}"
            else:
                comment = ", ".join([f"{v} (0x{v:x})" for v in field.allowed_values])
            default = hex(field.reset_content)

    else:
        value = str(field.content)
        if isinstance(field.allowed_values, range):
            comment = f"0..{field.allowed_values.stop - 1}"
        else:
            comment = ", ".join([str(v) for v in field.allowed_values])
        default = str(field.reset_content)

    assignment = f"{field.name} = {value}"

    if comments:
        assignment_with_comment = (
            f"{assignment:<{TOML_FIELD_COMMENT_START_AT_COLUMN}}# {comment}"
        )

        return (
            f"{assignment_with_comment:<{TOML_FIELD_DEFAULT_VALUE_START_AT}}"
            f" Reset: {default}"
        )

    return assignment


def register_as_toml(register: Register, **kwargs) -> str:
    """
    TOML representation of a peripheral register.

    :kwarg reset_values: Include unmodified Fields as their reset value
    :kwarg comments: Allow field comment strings in the TOML
    :kwarg force_32_bit_fields: List of Fields that should be given as 32-bit integers, regardless
        of bit width (for example, addresses).

    :return: TOML string representation of the register contents.
    """
    comments = kwargs.get("comments")
    reset_values = kwargs.get("reset_values")
    force_32_bit_fields = kwargs.get("force_32_bit_fields")

    return "\n".join(
        [
            field_as_toml(
                field, comments=comments, force_32_bit_fields=force_32_bit_fields
            )
            for field in register.values()
            if (reset_values or field.modified)
        ]
    )


def generate_toml(out_path: Path, peripheral: LogicalPeripheral, **kwargs):
    """
    Write the peripheral representation to a TOML file.

    This is written without the use of 3rd party libraries, for three reasons:
        1) Most TOML libraries focus on reading, rather than writing.
        2) The libraries that do support writing, do not always handle comments.
        3) Writing TOML is simple.

    :param out_path: Path to write TOML file to.
    :param peripheral: Peripheral to map register content into TOML

    :kwarg reset_values: Include register bit reset values along with modified values.
    :kwarg comments: Include register field comments in the output configuration
    """

    reset_values = kwargs.get("reset_values")
    comments = kwargs.get("comments")

    peripheral_toml: List[str] = []

    if peripheral.name.upper() == "UICR":
        force_32_bit_fields = uicr.FORCE_32_BIT_FIELDS
    else:
        force_32_bit_fields = []

    for _address, register in sorted(peripheral.address_to_register.items()):
        if reset_values or register.modified:
            register_toml = register_as_toml(
                register,
                reset_values=reset_values,
                comments=comments,
                force_32_bit_fields=force_32_bit_fields,
            )
            peripheral_toml.append(
                f"[{peripheral.path_to_record_name[register.path]}]\n{register_toml}"
            )

    file_content = "\n\n".join(peripheral_toml)
    trailing_newline = "\n" if len(file_content) != 0 else ""

    with open(out_path, "w", encoding="utf-8") as file:
        file.write(file_content + trailing_newline)


def generate_hex(out_path: Path, peripheral: LogicalPeripheral, **kwargs):
    """
    Write the peripheral representation to a HEX file.

    :param out_path: Path to write HEX file to.
    :param peripheral: Peripheral to map register content into hex

    :kwarg reset_values: Include register bit reset values along with modified values.
    :kwarg fill_byte: Byte value used to fill unused memory in the peripheral space.
    :kwarg fill: Method for writing fill byte to memory.
    """

    reset_values = kwargs.get("reset_values")
    fill_byte = kwargs.get("fill_byte")
    fill_type = kwargs.get("fill")

    raw_memory: Dict[int, int] = dict(
        peripheral.memory_map if reset_values else peripheral.written_memory_map
    )

    if fill_byte is not None:
        if fill_type == "unmodified":
            # Write the fill byte to all addresses that correspond to a register but that hasn't
            # been modified
            unmodified_addresses = (
                peripheral.memory_map.keys() - peripheral.written_memory_map.keys()
            )
            raw_memory.update({address: fill_byte for address in unmodified_addresses})

        elif fill_type == "all":
            # Write the fill byte to every unmodified address in the peripheral's address range,
            # even those that don't correspond to a register
            gap_addresses = (
                set(chain.from_iterable(peripheral.address_ranges)) - raw_memory.keys()
            )
            raw_memory.update({address: fill_byte for address in gap_addresses})

    ih = IntelHex(raw_memory)
    ih.write_hex_file(out_path)


def split_hex_diffs(
    base_hex: Path,
    min_delta: int = MIN_UICR_DIFF_DELTA,
    filled: bool = False,
    split_count: Optional[int] = None,
) -> List[IntelHex]:
    """
    Split a hex object into multiple hex objects based on diffs of a specific delta in addresses.

    :kwarg split_count: Specific number of splits to create.
    :kwarg min_delta: Minimum address delta between diffs. Reduce this to increase the number of diffs.
    :kwarg filled: Hex has been filled with a byte value and requires special handling for determining
                   diff boundaries.

    :return: Dictionary of IntelHex objects representing the split hexes.
    """

    in_hex = IntelHex()
    in_hex.loadhex(base_hex)

    addresses = sorted(in_hex.addresses())
    enumerated_pair = more.pairwise(enumerate(addresses))

    diffs = [
        (i + 1, addr_b - addr_a)
        for (i, addr_a), (_, addr_b) in enumerated_pair
        if (addr_b - addr_a) > min_delta
    ]
    diffs.sort(key=lambda x: x[1], reverse=True)

    if split_count is None:
        # Filled hexes can have one huge diff between two filled regions, with all other diffs being
        # 1 due to fill byte range differences. This results in one massive hex instead of one with
        # a proper split. This can be accounted for by incrementing the enumerated splits by 1 so
        # that the enumeration aligns with the enumeration of split bounds.
        split_count = len(diffs) + 1 if filled else len(diffs)

    if split_count <= 1:
        return [in_hex]

    max_diffs = diffs[: split_count - 1]

    # pylint: disable=unbalanced-tuple-unpacking
    splits, _ = more.unzip(max_diffs)

    split_bounds = [0, *sorted(splits), len(addresses)]
    split_hexes = []

    for start, end in more.pairwise(split_bounds):
        split_hexes.append(_make_hex_segment(in_hex, addresses[start:end]))

    return split_hexes


def _make_hex_segment(in_hex: IntelHex, addresses: List[int]) -> IntelHex:
    """Create a new hex file from a segment of an existing hex file."""
    new_hex = IntelHex()
    for address in addresses:
        new_hex[address] = in_hex[address]
    return new_hex
