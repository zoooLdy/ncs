#
# Copyright (c) 2024 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#
import dataclasses as dc
import enum
from typing import Optional, Tuple, Union

from devicetree import edtlib

from nrfregtool.common import NrfPsel, log_dbg


@dc.dataclass(frozen=True)
class GpiosProp:
    """CTRLSEL lookup table entry for *-gpios properties"""

    name: str
    port: int
    pin: int


@enum.unique
class Ctrlsel(enum.IntEnum):
    """
    Enumeration of GPIO.PIN_CNF[n].CTRLSEL values.
    The list here may not be exhaustive.
    """

    GPIO = 0
    VPR_GRC = 1
    CAN_PWM_I3C = 2
    SERIAL0 = 3
    EXMIF_RADIO_SERIAL1 = 4
    CAN_SERIAL2 = 5
    CAN = 6
    TND = 7


# Default CTRLSEL value indicating that CTRLSEL should not be used
CTRLSEL_DEFAULT = Ctrlsel.GPIO

# Pin functions used with pinctrl, see include/zephyr/dt-bindings/pinctrl/nrf-pinctrl.h
# Only the functions relevant for CTRLSEL deduction have been included.
NRF_FUN_UART_TX = 0
NRF_FUN_UART_RX = 1
NRF_FUN_UART_RTS = 2
NRF_FUN_UART_CTS = 3
NRF_FUN_SPIM_SCK = 4
NRF_FUN_SPIM_MOSI = 5
NRF_FUN_SPIM_MISO = 6
NRF_FUN_SPIS_SCK = 7
NRF_FUN_SPIS_MOSI = 8
NRF_FUN_SPIS_MISO = 9
NRF_FUN_SPIS_CSN = 10
NRF_FUN_TWIM_SCL = 11
NRF_FUN_TWIM_SDA = 12
NRF_FUN_PWM_OUT0 = 22
NRF_FUN_PWM_OUT1 = 23
NRF_FUN_PWM_OUT2 = 24
NRF_FUN_PWM_OUT3 = 25
NRF_FUN_EXMIF_CK = 35
NRF_FUN_EXMIF_DQ0 = 36
NRF_FUN_EXMIF_DQ1 = 37
NRF_FUN_EXMIF_DQ2 = 38
NRF_FUN_EXMIF_DQ3 = 39
NRF_FUN_EXMIF_DQ4 = 40
NRF_FUN_EXMIF_DQ5 = 41
NRF_FUN_EXMIF_DQ6 = 42
NRF_FUN_EXMIF_DQ7 = 43
NRF_FUN_EXMIF_CS0 = 44
NRF_FUN_EXMIF_CS1 = 45
NRF_FUN_CAN_TX = 46
NRF_FUN_CAN_RX = 47


_PINCTRL_CTRLSEL_LOOKUP_NRF54H20 = {
    # CAN120
    0x5F8D_8000: {
        # P2
        NrfPsel(fun=NRF_FUN_CAN_TX, port=2, pin=9): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_CAN_RX, port=2, pin=8): Ctrlsel.CAN_PWM_I3C,
        # P9
        NrfPsel(fun=NRF_FUN_CAN_TX, port=9, pin=5): Ctrlsel.CAN,
        NrfPsel(fun=NRF_FUN_CAN_RX, port=9, pin=4): Ctrlsel.CAN,
    },
    # PWM120
    0x5F8E_4000: {
        # P2
        NrfPsel(fun=NRF_FUN_PWM_OUT0, port=2, pin=4): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT1, port=2, pin=5): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT2, port=2, pin=6): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT3, port=2, pin=7): Ctrlsel.CAN_PWM_I3C,
        # P6
        NrfPsel(fun=NRF_FUN_PWM_OUT0, port=6, pin=6): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT1, port=6, pin=7): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT2, port=6, pin=8): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT3, port=6, pin=9): Ctrlsel.CAN_PWM_I3C,
        # P7
        NrfPsel(fun=NRF_FUN_PWM_OUT0, port=7, pin=0): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT1, port=7, pin=1): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT2, port=7, pin=6): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT3, port=7, pin=7): Ctrlsel.CAN_PWM_I3C,
    },
    # PWM130
    0x5F9A_4000: {
        # P9
        NrfPsel(fun=NRF_FUN_PWM_OUT0, port=9, pin=2): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT1, port=9, pin=3): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT2, port=9, pin=4): Ctrlsel.CAN_PWM_I3C,
        NrfPsel(fun=NRF_FUN_PWM_OUT3, port=9, pin=5): Ctrlsel.CAN_PWM_I3C,
    },
    # SPIM130/SPIS130/TWIM130/TWIS130/UARTE130
    0x5F9A_5000: {
        # SPIM mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=9, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=9, pin=2): Ctrlsel.SERIAL0,
        GpiosProp(name="cs-gpios", port=9, pin=3): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=9, pin=4): Ctrlsel.SERIAL0,
        # SPIS mappings
        NrfPsel(fun=NRF_FUN_SPIS_MISO, port=9, pin=2): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIS_MOSI, port=9, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIS_CSN, port=9, pin=3): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIS_SCK, port=9, pin=4): Ctrlsel.SERIAL0,
        # TWIM mappings
        NrfPsel(fun=NRF_FUN_TWIM_SDA, port=9, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_TWIM_SCL, port=9, pin=4): Ctrlsel.SERIAL0,
        # UARTÈ mappings
        NrfPsel(fun=NRF_FUN_UART_TX, port=9, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RX, port=9, pin=4): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_UART_CTS, port=9, pin=2): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RTS, port=9, pin=3): Ctrlsel.SERIAL0,
    },
    # SPIM131/SPIS131/TWIM131/TWIS131/UARTE131
    0x5F9A_6000: {
        # SPIM mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=9, pin=0): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=9, pin=2): Ctrlsel.CAN_SERIAL2,
        GpiosProp(name="cs-gpios", port=9, pin=3): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=9, pin=1): Ctrlsel.CAN_SERIAL2,
        # SPIS mappings
        NrfPsel(fun=NRF_FUN_SPIS_MISO, port=9, pin=0): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_SPIS_MOSI, port=9, pin=2): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_SPIS_CSN, port=9, pin=3): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_SPIS_SCK, port=9, pin=1): Ctrlsel.CAN_SERIAL2,
        # TWIM mappings
        NrfPsel(fun=NRF_FUN_TWIM_SDA, port=9, pin=0): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_TWIM_SCL, port=9, pin=1): Ctrlsel.CAN_SERIAL2,
        # UARTÈ mappings
        NrfPsel(fun=NRF_FUN_UART_TX, port=9, pin=0): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_UART_RX, port=9, pin=1): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_CTS, port=9, pin=2): Ctrlsel.CAN_SERIAL2,
        NrfPsel(fun=NRF_FUN_UART_RTS, port=9, pin=3): Ctrlsel.CAN_SERIAL2,
    },
    # SPIM120/SPIS120/UARTE120
    0x5F8E_6000: {
        # SPIM P6 mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=6, pin=8): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=6, pin=7): Ctrlsel.SERIAL0,
        GpiosProp(name="cs-gpios", port=6, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=6, pin=1): Ctrlsel.SERIAL0,
        # SPIM P7 mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=7, pin=7): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=7, pin=6): Ctrlsel.SERIAL0,
        GpiosProp(name="cs-gpios", port=7, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=7, pin=3): Ctrlsel.SERIAL0,
        # SPIM P2 mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=2, pin=6): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=2, pin=5): Ctrlsel.SERIAL0,
        GpiosProp(name="cs-gpios", port=2, pin=7): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=2, pin=3): Ctrlsel.SERIAL0,
        # SPIS P6 mappings
        NrfPsel(fun=NRF_FUN_SPIS_MISO, port=6, pin=3): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIS_MOSI, port=6, pin=4): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIS_CSN, port=6, pin=9): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIS_SCK, port=6, pin=0): Ctrlsel.SERIAL0,
        # UARTÈ P6 mappings
        NrfPsel(fun=NRF_FUN_UART_TX, port=6, pin=8): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_CTS, port=6, pin=7): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RX, port=6, pin=6): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RTS, port=6, pin=5): Ctrlsel.SERIAL0,
        # UARTÈ P7 mappings
        NrfPsel(fun=NRF_FUN_UART_TX, port=7, pin=7): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_CTS, port=7, pin=6): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RX, port=7, pin=4): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RTS, port=7, pin=5): Ctrlsel.SERIAL0,
        # UARTÈ P2 mappings
        NrfPsel(fun=NRF_FUN_UART_TX, port=2, pin=6): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_CTS, port=2, pin=5): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RX, port=2, pin=4): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_UART_RTS, port=2, pin=7): Ctrlsel.SERIAL0,
    },
    # SPIM121
    0x5F8E_7000: {
        # SPIM P6 mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=6, pin=13): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=6, pin=12): Ctrlsel.SERIAL0,
        GpiosProp(name="cs-gpios", port=6, pin=10): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=6, pin=2): Ctrlsel.SERIAL0,
        # SPIM P7 mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=7, pin=1): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=7, pin=0): Ctrlsel.EXMIF_RADIO_SERIAL1,
        GpiosProp(name="cs-gpios", port=7, pin=4): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=7, pin=2): Ctrlsel.EXMIF_RADIO_SERIAL1,
        # SPIM P2 mappings
        NrfPsel(fun=NRF_FUN_SPIM_MOSI, port=2, pin=11): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_MISO, port=2, pin=10): Ctrlsel.SERIAL0,
        GpiosProp(name="cs-gpios", port=2, pin=8): Ctrlsel.SERIAL0,
        NrfPsel(fun=NRF_FUN_SPIM_SCK, port=2, pin=2): Ctrlsel.SERIAL0,
    },
    # EXMIF
    0x5F09_5000: {
        NrfPsel(fun=NRF_FUN_EXMIF_CK, port=6, pin=0): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ7, port=6, pin=4): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ1, port=6, pin=5): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ6, port=6, pin=6): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ0, port=6, pin=7): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ5, port=6, pin=8): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ3, port=6, pin=9): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ2, port=6, pin=10): Ctrlsel.EXMIF_RADIO_SERIAL1,
        NrfPsel(fun=NRF_FUN_EXMIF_DQ4, port=6, pin=11): Ctrlsel.EXMIF_RADIO_SERIAL1,
        # Ref: NCSDK-27559
        NrfPsel(fun=NRF_FUN_EXMIF_CS0, port=6, pin=3): Ctrlsel.GPIO,
        GpiosProp(name="cs-gpios", port=6, pin=3): Ctrlsel.GPIO,
        NrfPsel(fun=NRF_FUN_EXMIF_CS1, port=6, pin=13): Ctrlsel.GPIO,
        GpiosProp(name="cs-gpios", port=6, pin=13): Ctrlsel.GPIO,
    },
}


def dt_lookup_ctrlsel(
    src: Union[edtlib.PinCtrl, edtlib.Property],
    psel: Union[NrfPsel, Tuple[int, int]],
) -> Optional[int]:
    """Get the CTRLSEL value to use for a given pin selection."""

    root_compatible = src.node.edt.get_node("/").compats[0]
    if root_compatible.startswith("nordic,nrf54h20"):
        lut = _PINCTRL_CTRLSEL_LOOKUP_NRF54H20
    else:
        return None

    if isinstance(src, edtlib.PinCtrl):
        identifier = src.node.regs[0].addr
        sub_entry = psel
    elif isinstance(src, edtlib.Property):
        try:
            identifier = src.node.regs[0].addr
        except IndexError:
            identifier = src.node.label
        sub_entry = GpiosProp(name=src.name, port=psel[0], pin=psel[1])
    else:
        raise ValueError(f"Unsupported GPIO pin source: {src}")

    if identifier in lut:
        ctrlsel = lut[identifier].get(sub_entry, CTRLSEL_DEFAULT)
    else:
        ctrlsel = None

    log_dbg(
        f"identifier={hex(identifier) if isinstance(identifier, int) else identifier}, "
        f"{sub_entry=} -> {ctrlsel=}"
    )

    return ctrlsel
