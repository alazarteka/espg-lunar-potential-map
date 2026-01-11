from typing import Annotated, Any

import pint
from pint import Quantity

ureg = pint.UnitRegistry()
# ureg.enable_contexts("numpy")


def validate_quantity(value: Any, expected_unit: pint.Unit, name: str) -> Quantity:
    if not isinstance(value, Quantity) or not value.is_compatible_with(expected_unit):
        raise TypeError(
            f"{name} must be a pint Quantity compatible with {expected_unit}"
        )
    return value


LengthType = Annotated[Quantity, ureg.meter]
SpeedType = Annotated[Quantity, ureg.meter / ureg.second]
MassType = Annotated[Quantity, ureg.kilogram]
EnergyType = Annotated[Quantity, ureg.joule]
NumberDensityType = Annotated[Quantity, ureg.particle / (ureg.meter**3)]
DimensionlessType = Annotated[Quantity, ureg.dimensionless]
ChargeType = Annotated[Quantity, ureg.coulomb]
TimeType = Annotated[Quantity, ureg.second]
VoltageType = Annotated[Quantity, ureg.volt]
AngleType = Annotated[Quantity, ureg.degree]
FluxType = Annotated[
    Quantity,
    ureg.particle
    / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt),
]
OmnidirectionalFluxType = Annotated[
    Quantity,
    ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt),
]
IntegratedFluxType = Annotated[
    Quantity, ureg.particle / (ureg.centimeter**2 * ureg.second)
]
PhaseSpaceDensityType = Annotated[
    Quantity, ureg.particle / (ureg.meter**3 * (ureg.meter / ureg.second) ** 3)
]
CurrentDensityType = Annotated[Quantity, ureg.ampere / ureg.meter**2]

__all__ = [
    "AngleType",
    "ChargeType",
    "CurrentDensityType",
    "DimensionlessType",
    "EnergyType",
    "FluxType",
    "IntegratedFluxType",
    "LengthType",
    "MassType",
    "NumberDensityType",
    "OmnidirectionalFluxType",
    "PhaseSpaceDensityType",
    "SpeedType",
    "TimeType",
    "VoltageType",
    "ureg",
    "validate_quantity",
]
