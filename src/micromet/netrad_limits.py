import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from math import sin, cos, radians, pi, asin

# Constants
SOLAR_CONSTANT = 1367  # W/m²
ALBEDO = 0.25  # average for natural terrain
EMISSIVITY = 0.98  # ground emissivity
STEFAN_BOLTZMANN = 5.67e-8  # W/m²/K⁴
LATITUDE = 39.5  # Utah average latitude in degrees


def solar_declination(doy):
    """Return solar declination angle in radians for a given day of year."""
    return radians(23.44) * sin(2 * pi * (284 + doy) / 365)


def hour_angle(hour):
    """Return hour angle in radians."""
    return radians(15 * (hour - 12))


def solar_elevation(doy, hour, latitude=LATITUDE):
    """Return solar elevation angle in degrees."""
    decl = solar_declination(doy)
    lat_rad = radians(latitude)
    ha = hour_angle(hour)

    sin_elev = sin(lat_rad) * sin(decl) + cos(lat_rad) * cos(decl) * cos(ha)
    return max(0, asin(sin_elev))  # radians


def clear_sky_radiation(doy, hour, latitude=LATITUDE):
    """Estimate incoming shortwave radiation under clear-sky conditions."""
    elev_rad = solar_elevation(doy, hour, latitude)
    if elev_rad <= 0:
        return 0.0

    d_r = 1 + 0.033 * cos(2 * pi * doy / 365)  # Earth-Sun distance factor
    cos_zenith = cos(pi / 2 - elev_rad)
    rs = SOLAR_CONSTANT * d_r * cos_zenith * 0.75  # ~0.75 = clear sky transmissivity
    return rs


def longwave_radiation(T_kelvin):
    """Estimate longwave radiation using Stefan-Boltzmann law."""
    return EMISSIVITY * STEFAN_BOLTZMANN * T_kelvin**4


def estimate_net_radiation_range(doy, hour):
    """Estimate min/max net radiation for given hour and DOY in Utah."""
    rs_down = clear_sky_radiation(doy, hour)

    # Net shortwave
    rs_net = rs_down * (1 - ALBEDO)

    # Assume typical diurnal surface temperature range in K
    Tmin_K = 273.15 + 5  # typical min in early morning
    Tmax_K = 273.15 + 40  # hot afternoon summer temp

    # Incoming longwave (simplified as ~cloudless sky)
    lw_down_min = longwave_radiation(Tmin_K - 5)  # colder sky at night
    lw_down_max = longwave_radiation(Tmax_K - 15)  # warmer sky in afternoon

    # Outgoing longwave from surface
    lw_up_min = longwave_radiation(Tmin_K)
    lw_up_max = longwave_radiation(Tmax_K)

    # Net radiation range
    Rn_min = rs_net + lw_down_min - lw_up_max
    Rn_max = rs_net + lw_down_max - lw_up_min

    return Rn_min, Rn_max


def add_buffer(min_max: tuple, buffer: float = 100):
    if min_max[0] - buffer <= -200:
        minv = -200
    else:
        minv = min_max[0] - buffer
    maxv = min_max[1] + buffer
    return minv, maxv


if __name__ == "__main__":
    # Example usage
    doy = 172  # summer solstice ~June 21
    hour = 14  # 2 PM

    Rn_min, Rn_max = estimate_net_radiation_range(doy, hour)
    print(f"DOY: {doy}, Hour: {hour}")
    print(f"Estimated Net Radiation Range: {Rn_min:.1f} W/m² to {Rn_max:.1f} W/m²")

    def calc_diurnal_range(doy):
        hours = np.arange(0, 24)
        rn_min_list, rn_max_list = zip(
            *[estimate_net_radiation_range(doy, h) for h in hours]
        )
        return rn_min_list, rn_max_list
