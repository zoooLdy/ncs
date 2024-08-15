#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import dataclasses as dc
import enum
from collections.abc import Mapping
from typing import Any, Dict, Sequence, Union

import click

_LOG_VERBOSITY = 0


@enum.unique
class AddressOffset(enum.IntEnum):
    """
    Enumeration of address bit offsets, defined by Address Format of the product specification.
    """

    REGION = 29
    SECURITY = 28
    DOMAINID = 24
    ADDR = 23


class _HaltiumID(enum.Enum):
    """
    Common parent class of Haltium ID enums.
    """

    def __int__(self):
        """Integer representation of the class."""
        return int(self.value)

    def __lt__(self, other):
        """Overridden less-than to guarantee check of Enum int values."""
        if self.value < int(other):
            return True
        return False

    def __str__(self):
        """String representation of the class."""
        return self.name

    @classmethod
    def from_value(cls, value: int):
        """
        Helper method to return the enum object equivalent to an integer, if it is enumerated.
        """
        for x in cls:
            if int(x) == value:
                return x
        return None

    def id_string(self):
        return f"{self.value} ({self.name})"


@enum.unique
class AddressRegion(_HaltiumID):
    """
    Enumeration of address regions, defined by Address Format of the product specification.
    """

    PROGRAM = 0
    DATA = 1
    PERIPHERAL = 2
    EXT_XIP = 3
    EXT_XIP_ENCRYPTED = 4
    STM = 5
    CPU = 7

    @classmethod
    def from_address(cls, address: int) -> AddressRegion:
        """
        Helper method to extract an enumerated address region from a given address.
        """
        return cls.from_value((address >> AddressOffset.REGION) & 0b111)


@enum.unique
class DomainID(_HaltiumID):
    """
    Enumeration of domain IDs in Haltium products.
    """

    SECURE = 1
    APPLICATION = 2
    RADIOCORE = 3
    GLOBAL_FAST = 12
    GLOBAL_SLOW = 13
    GLOBAL_ = 14
    GLOBAL = 15

    @classmethod
    def from_address(cls, address: int):
        """
        Helper method to extract an enumerated domain ID from a given address.
        """
        return cls.from_value((address >> AddressOffset.DOMAINID) & 0xF)

    @classmethod
    def from_processor(cls, processor: Union[ProcessorID, int]):
        """
        Helper method to extract an enumerated domain ID from a processor ID.
        """
        processor_domain = {
            ProcessorID.SECURE.value: cls.SECURE,
            ProcessorID.APPLICATION.value: cls.APPLICATION,
            ProcessorID.RADIOCORE.value: cls.RADIOCORE,
            ProcessorID.SYSCTRL.value: cls.GLOBAL_FAST,
            ProcessorID.PPR.value: cls.GLOBAL_SLOW,
            ProcessorID.FLPR.value: cls.GLOBAL_,
        }
        return processor_domain.get(int(processor))


@enum.unique
class OwnerID(_HaltiumID):
    """
    Enumeration of ownership IDs in haltium products.
    """

    NONE = 0
    SECURE = 1
    APPLICATION = 2
    RADIOCORE = 3
    SYSCTRL = 8

    @classmethod
    def from_domain(cls, domain: DomainID):
        """
        Helper method to extract an enumerated owner ID from a domain ID.
        """
        domain_owner = {
            DomainID.SECURE.value: cls.SECURE,
            DomainID.APPLICATION.value: cls.APPLICATION,
            DomainID.RADIOCORE.value: cls.RADIOCORE,
            DomainID.GLOBAL_FAST.value: cls.SYSCTRL,
        }
        return domain_owner.get(int(domain))

    @classmethod
    def from_processor(cls, processor: Union[ProcessorID, int]):
        """
        Helper method to extract an enumerated owner ID from a processor ID.
        """
        if (domain := DomainID.from_processor(processor)) is not None:
            return cls.from_domain(domain)
        return None


@enum.unique
class ProcessorID(_HaltiumID):
    """
    Enumeration of processor IDs in haltium products.
    """

    SECURE = 1
    APPLICATION = 2
    RADIOCORE = 3
    BBPR = 11
    SYSCTRL = 12
    PPR = 13
    FLPR = 14

    @classmethod
    def from_domain(cls, domain: DomainID):
        """
        Helper method to extract an enumerated processor ID from a domain ID.
        """
        domain_processor = {
            DomainID.SECURE.value: cls.SECURE,
            DomainID.APPLICATION.value: cls.APPLICATION,
            DomainID.RADIOCORE.value: cls.RADIOCORE,
            DomainID.GLOBAL_FAST.value: cls.SYSCTRL,
            DomainID.GLOBAL_SLOW.value: cls.PPR,
            DomainID.GLOBAL_.value: cls.FLPR,
        }
        return domain_processor.get(int(domain))

    @classmethod
    def from_nodelabels(cls, labels: List[str]):
        """
        Helper method to extract an enumerated processor ID from a list of devicetree nodelabels.
        """
        substring_processor = {
            "cpusec": cls.SECURE,
            "cpuapp": cls.APPLICATION,
            "cpurad": cls.RADIOCORE,
            "cpubbpr": cls.BBPR,
            "cpusys": cls.SYSCTRL,
            "cpuppr": cls.PPR,
            "cpuflpr": cls.FLPR,
        }
        processors = {
            processor_id
            for substring, processor_id in substring_processor.items()
            if any(substring in label for label in labels)
        }
        if len(processors) == 1:
            return processors.pop()
        return None


# Bit position of the function bits in the pinctrl pin value encoded from NRF_PSEL()
NRF_PSEL_FUN_POS = 17
# Mask for the function bits in the pinctrl pin value encoded from NRF_PSEL()
NRF_PSEL_FUN_MASK = 0x7FFF << NRF_PSEL_FUN_POS
# Pull this from DT
GPIO_PIN_COUNT = 32


@dc.dataclass(frozen=True)
class NrfPsel:
    """Decoded NRF_PSEL values."""

    fun: int
    port: int
    pin: int


def nrf_psel_decode(psel_value: int) -> NrfPsel:
    """
    Decode an NRF_PSEL encoded GPIO pin value to its individual parts.

    :param psel_value: GPIO pin value encoded with NRF_PSEL()
    :return: A decoded tuple of GPIO (port, pin)
    """
    port, pin = gpio_port_pin_decode(psel_value & (~NRF_PSEL_FUN_MASK))
    fun = (psel_value & NRF_PSEL_FUN_MASK) >> NRF_PSEL_FUN_POS
    return NrfPsel(fun=fun, port=port, pin=pin)


def gpio_port_pin_decode(port_pin_val: int) -> Tuple[int, int]:
    """
    Decode a GPIO pin value to a tuple consisting of (port, pin)

    :param port_pin_val: GPIO pin value containing an encoded port and pin number
    :return: A decoded tuple of GPIO (port, pin)
    """
    return (port_pin_val // GPIO_PIN_COUNT, port_pin_val % GPIO_PIN_COUNT)


class Record(Dict[str, Dict[str, Union[str, int]]]):
    """
    A transcript of the contents of any number of registers and their fields.
    A basic Record is one that is not complete with register instance information, while a complete
    Record is one that also carries instance information.

    Basic example: "mem_config"
    Complete example: "mem_config_0"

    Example Format:
    record = {
        "mem_config_0": {
            "READ": "Allowed",
            "WRITE": "Allowed",
            ...
        }
        ...
    }
    """

    def update(self, other=(), /, **kwargs):
        """
        Overridden method. If a Record is given as the parameter and already has an entry in the
        mapping, update the Record contents instead. This avoids overwriting the Record entirely
        when it should instead be patched or updated as the script processes all input contents.
        If no matching Record exists or the argument is not a Record object, the parent method
        functionality is used.

        :param other: Positional arguments. See `dict.update` for more details.
        :param kwargs: Keyword arguments. See `dict.update` for more details.
        """

        if isinstance(other, Mapping):
            for key, value in other.items():
                if isinstance(self.get(key), Mapping) and isinstance(value, Mapping):
                    self[key].update(value)
                else:
                    self[key] = value
        else:
            super().update(other)

        for key, value in kwargs.items():
            if isinstance(self.get(key), Mapping) and isinstance(value, Mapping):
                self[key].update(value)
            else:
                self[key] = value

    def as_hex(self):
        """
        Helper to replace int members with their value as a hex string.
        """

        def int_to_hex(root: dict) -> Record:
            """Internal helper function for traversing the nested maps (recursive)."""

            clone = {}
            for k, v in root.items():
                if isinstance(v, int):
                    clone[k] = hex(v)
                elif isinstance(v, dict):
                    clone[k] = int_to_hex(v)
                else:
                    clone[k] = v
            return clone

        return int_to_hex(self)


def config_log(**kwargs):
    """
    Configure logging settings.

    :kwarg verbosity: Set level of verbose output
    """
    verbosity = kwargs.get("verbosity")

    if verbosity is not None:
        global _LOG_VERBOSITY
        _LOG_VERBOSITY = int(verbosity)


def log_inf(msg: str):
    """
    Print information.

    :param msg: Message to print
    """
    if _LOG_VERBOSITY >= 1:
        click.echo(msg)


def log_vrb(msg: str):
    """
    Print verbose information.

    :param msg: Message to print
    """
    if verbose_logs():
        click.secho(msg, fg="green")


def log_dbg(msg: str):
    """
    Print debug information.

    :param msg: Message to print
    """
    if _LOG_VERBOSITY >= 2:
        click.secho(msg, fg="blue")


def log_dev(msg: str):
    """
    Print detailed information for development.

    :param msg: Message to print
    """
    if _LOG_VERBOSITY >= 3:
        click.secho(msg, fg="magenta")


def log_err(msg: str):
    """
    Print error information.

    :param msg: Message to print
    """
    click.secho(f"ERROR: {msg}", err=True, fg="red")


def verbose_logs():
    """
    :return: True if verbosity is at or above the Verbose level.
    """
    return _LOG_VERBOSITY >= 1


def else_none(obj: Any, text=None) -> Union[Any, str]:
    """
    Helper function for printing an empty object nicely instead of an empty or None.

    :param obj: Object to check against being None or empty
    :kwarg text: Specific text to return

    :return: Pretty help text
    """

    if text is None and isinstance(obj, Sequence):
        text = "Empty"

    return obj if obj else str(text)
