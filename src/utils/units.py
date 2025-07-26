from typing import Annotated

import pint
from pint import Quantity

ureg = pint.UnitRegistry()
# ureg.enable_contexts("numpy")

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


__all__ = [
    "ureg",
    "LengthType",
    "SpeedType",
    "MassType",
    "EnergyType",
    "NumberDensityType",
    "DimensionlessType",
    "ChargeType",
    "TimeType",
    "VoltageType",
    "AngleType",
    "FluxType",
    "OmnidirectionalFluxType",
    "IntegratedFluxType",
    "PhaseSpaceDensityType",
]
