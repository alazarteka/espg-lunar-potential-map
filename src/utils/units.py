from typing import Annotated

import pint
from pint import Quantity

ureg = pint.UnitRegistry()
# ureg.enable_contexts("numpy")

Length = Annotated[Quantity, ureg.meter]
Speed = Annotated[Quantity, ureg.meter / ureg.second]
Mass = Annotated[Quantity, ureg.kilogram]
Energy = Annotated[Quantity, ureg.joule]
NumberDensity = Annotated[Quantity, ureg.particle / (ureg.meter ** 3)]
Dimensionless = Annotated[Quantity, ureg.dimensionless]
Charge = Annotated[Quantity, ureg.coulomb]
Time = Annotated[Quantity, ureg.second]
Voltage = Annotated[Quantity, ureg.volt]
Angle = Annotated[Quantity, ureg.degree]
Flux = Annotated[Quantity, ureg.particle / (ureg.centimeter ** 2 * ureg.second * ureg.steradian * ureg.electron_volt)]
PhaseSpaceDensity = Annotated[Quantity, ureg.particle / (ureg.meter ** 3 * (ureg.meter / ureg.second) ** 3)]


__all__ = [
    "ureg",
    "Length",
    "Speed",
    "Mass",
    "Energy",
    "NumberDensity",
    "Dimensionless",
    "Charge",
    "Time",
    "Voltage",
    "Angle",
    "Flux",
    "PhaseSpaceDensity"
]
