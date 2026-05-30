import sys
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy import interpolate
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import UnivariateSpline
import astropy
from astropy.coordinates import EarthLocation, get_sun, AltAz
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u
import csv
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from numpy import exp,arange
import matplotlib.pyplot as plt

iers.conf.auto_download = False

"""
calculate the local sidereal time (LAST) 
time is relative to midnight GMT 
arXiv:1312.1355
"""
def FracDays(date, time:np.array([0,0,0])):
    """
    fractional day number
    the zero of reference is 1 Jan 2000
    """
    n = 0
    
    d = date[0]
    m = date[1]
    y = date[2]
    
    h = time[0]
    minute = time[1]
    sec = time[2]
    
    if m == 1 or m == 2:
        Ytilde = y-1
        Mtilde = m+12
    else: 
        Ytilde = y
        Mtilde = m
    
    n += np.floor(365.25*Ytilde)+np.floor(30.61*(Mtilde+1))+d-730563.5+h/24+minute/(24*60)+sec/(24*60*60)
    return n
def GMST(n):
    T = n/36525
    return (86400*(0.7790572732640+0.00273781191135448*n+n%1)+0.00096707+307.47710227*T+0.092772113*T*T)%86400

def EqEq(n):
    T = n/36525
    Omega = np.deg2rad(125.04455501)-n*np.deg2rad(0.05295376)
    L=np.deg2rad(280.47)+n*np.deg2rad(0.98565)
    Dpsi = -1.1484*np.sin(Omega)-0.0864*np.sin(2*L)
    eA = np.deg2rad(23.4392794444)-0.01301021361*T**2
    return Dpsi*np.cos(eA)+0.000176*np.sin(Omega)+0.000004*np.sin(2*Omega)

def GAST(n):
    return GMST(n)+EqEq(n)

def LAST(n,lon):
    return (GAST(n)+lon/(2*np.pi)*86400)%86400


"""
Coordinate transformations
"""
def HelioEcliptic2Galactic(vec,T:0):
    arcsecond = np.deg2rad(1/3600) # 1/3600 degrees
    zetaA = 2306.083227*arcsecond*T + 0.298850*arcsecond*T**2
    zA = 2306.077181*arcsecond*T + 1.092735*arcsecond*T**2
    thetaA = 2004.191903*arcsecond*T - 0.429493*arcsecond*T**2
    
    P = np.array([
        [np.cos(zetaA)*np.cos(thetaA)*np.cos(zA) - 
     np.sin(zetaA)*np.sin(zA), -np.sin(zetaA)* np.cos(thetaA)*np.cos(zA) - 
     np.cos(zetaA)*np.sin(zA), -np.sin(thetaA)*np.cos(zA)],
        [np.cos(zetaA)*np.cos(thetaA)*np.sin(zA) + 
     np.sin(zetaA)*np.cos(zA),-np.sin(zetaA)*np.cos(thetaA)*np.sin(zA) + 
     np.cos(zetaA)*np.cos(zA),-np.sin(thetaA)*np.sin(zA)],
        [np.cos(zetaA)*np.sin(thetaA), -np.sin(zetaA)*np.sin(thetaA), np.cos(thetaA)]
    ]) 
    
    lCP = np.deg2rad(122.932) # degree
    alphaGP = np.deg2rad(192.85948) # degree
    deltaGP = np.deg2rad(27.12825) # degree
    
    M = np.array([[-np.sin(lCP)*np.sin(alphaGP)-np.cos(lCP)*np.cos(alphaGP)*np.sin(deltaGP),np.sin(lCP)*np.cos(alphaGP)-np.cos(lCP)*np.sin(alphaGP)*np.sin(deltaGP),np.cos(lCP)*np.cos(deltaGP)],
                  [np.cos(lCP)*np.sin(alphaGP)-np.sin(lCP)*np.cos(alphaGP)*np.sin(deltaGP),-np.cos(lCP)*np.cos(alphaGP)-np.sin(lCP)*np.sin(alphaGP)*np.sin(deltaGP),np.sin(lCP)*np.cos(deltaGP)],
                  [np.cos(alphaGP)*np.cos(deltaGP),np.sin(alphaGP)*np.cos(deltaGP),np.sin(deltaGP)]
                 ]
    )
    epsilon = np.deg2rad(23.4393) - np.deg2rad(0.0130)*T
    R = np.array([[1,0,0],[0,np.cos(epsilon),-np.sin(epsilon)],[0,np.sin(epsilon),np.cos(epsilon)]])
    prod1=-np.dot(M,np.linalg.inv(P))
    prod2=np.dot(R,vec)
    return np.dot(prod1,prod2)

def Equatorial2Galactic(vec,T:0):
    arcsecond = np.deg2rad(1/3600) # 1/3600 degrees
    zetaA = 2306.083227*arcsecond*T + 0.298850*arcsecond*T**2
    zA = 2306.077181*arcsecond*T + 1.092735*arcsecond*T**2
    thetaA = 2004.191903*arcsecond*T - 0.429493*arcsecond*T**2
    
    P = np.array([
        [np.cos(zetaA)*np.cos(thetaA)*np.cos(zA) - 
     np.sin(zetaA)*np.sin(zA), -np.sin(zetaA)* np.cos(thetaA)*np.cos(zA) - 
     np.cos(zetaA)*np.sin(zA), -np.sin(thetaA)*np.cos(zA)],
        [np.cos(zetaA)*np.cos(thetaA)*np.sin(zA) + 
     np.sin(zetaA)*np.cos(zA),-np.sin(zetaA)*np.cos(thetaA)*np.sin(zA) + 
     np.cos(zetaA)*np.cos(zA),-np.sin(thetaA)*np.sin(zA)],
        [np.cos(zetaA)*np.sin(thetaA), -np.sin(zetaA)*np.sin(thetaA), np.cos(thetaA)]
    ]) 
    
    lCP = np.deg2rad(122.932) # degree
    alphaGP = np.deg2rad(192.85948) # degree
    deltaGP = np.deg2rad(27.12825) # degree
    
    M = np.array([[-np.sin(lCP)*np.sin(alphaGP)-np.cos(lCP)*np.cos(alphaGP)*np.sin(deltaGP),np.sin(lCP)*np.cos(alphaGP)-np.cos(lCP)*np.sin(alphaGP)*np.sin(deltaGP),np.cos(lCP)*np.cos(deltaGP)],
                  [np.cos(lCP)*np.sin(alphaGP)-np.sin(lCP)*np.cos(alphaGP)*np.sin(deltaGP),-np.cos(lCP)*np.cos(alphaGP)-np.sin(lCP)*np.sin(alphaGP)*np.sin(deltaGP),np.sin(lCP)*np.cos(deltaGP)],
                  [np.cos(alphaGP)*np.cos(deltaGP),np.sin(alphaGP)*np.cos(deltaGP),np.sin(deltaGP)]
                 ]
    )

    prod1 = np.dot(M,np.linalg.inv(P))
    return np.dot(prod1,vec)

"""
Earth's velocity
"""
def ve(n):
    vEavg = 29.79 # km/s
    vr = np.asarray([0,220,0]) # galactic rotation km/s
    vs = np.asarray([11.1,12.2,7.3]) # sun's relative motion km/s
    L=np.deg2rad(280.460)+np.deg2rad(0.9856474)*n # mean longitude
    ee = 0.01671 # ellipticity of Earth's orbit
    omega = np.deg2rad(282.932)+np.deg2rad(0.0000471)*n # perihilion longitude
    T=n/36525
    ex = np.array([1,0,0])
    ey = np.array([0,1,0])
    uE=-vEavg*(np.sin(L)+ee*np.sin(2*L-omega))*HelioEcliptic2Galactic(ex,T)+vEavg*(np.cos(L)+ee*np.cos(2*L-omega))*HelioEcliptic2Galactic(ey,T)
    return vr+vs+uE
"""
Laboratory position
"""
def LabPos(lat,lon,depth,n):
    r = 6371 - depth # km
    theta = np.pi/2-lat
    phi = 2*np.pi/86400*LAST(n,lon)
    coord = np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)])
    return Equatorial2Galactic(coord,n/36525)

def _quantity_to_value(quantity, unit):
    """Return a float from an Astropy Quantity or a plain numeric value."""
    if hasattr(quantity, "to_value"):
        return quantity.to_value(unit)
    return float(quantity)

def normalize_site_key(site):
    """Normalize supported site names and aliases to SITE_COORDS keys."""
    key = str(site).strip().upper()
    return SITE_ALIASES.get(key, key)

def normalize_thetaiso_loc(loc):
    """Return the (lat_rad, lon_rad, depth_km) tuple expected by ThetaIso."""
    if isinstance(loc, EarthLocation):
        return (
            loc.lat.to_value(u.rad),
            loc.lon.to_value(u.rad),
            -loc.height.to_value(u.km),
        )

    if isinstance(loc, str):
        return get_site_thetaiso_loc(loc)

    if isinstance(loc, dict) and "loc" in loc:
        loc = loc["loc"]

    lat = _quantity_to_value(loc[0], u.rad)
    lon = _quantity_to_value(loc[1], u.rad)
    depth = _quantity_to_value(loc[2], u.km)
    return (lat, lon, depth)

def ThetaIso(loc,n):
    """
    returns Isodetection angle in radians
    """
    loc = normalize_thetaiso_loc(loc)
    lat = loc[0]
    lon = loc[1]
    depth = loc[2]
    
    vector_1 = ve(n)
    vector_2 = LabPos(lat,lon,depth,n)
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

#Here is where locations are defined. Feel free to add more using the same format
SITE_COORDS = {
    # key: (lat_deg, lon_deg, height_m)
    "BRC": (-41.14557, -71.30822, 0),
    "SG": (-41.606, -65.355, 0),
    "FNAL": (41.82583, -88.25433, 226),
    "SNO": (46.4719, -81.201, 0),
    "GSSI": (42.45267, 13.5715, 963),
    "MODANE": (45.2, 6.69, 0),
    "SOUDAN": (47.8208333, -92.2361111, 481),
    "SUPL": (-36.060, 142.801, 0),
    "CAPETOWN": (-33.9249, 18.4241, 0),
}

SITE_ALIASES = {
    "SNOLAB": "SNO",
    "SNO LAB": "SNO",
    "SNOLAB, CANADA": "SNO",
    "SNO": "SNO",
    "BARILOCHE": "BRC",
    "SAN CARLOS DE BARILOCHE, ARGENTINA": "BRC",
    "FERMILAB": "FNAL",
    "FERMILAB, USA": "FNAL",
    "GRAN SASSO": "GSSI",
    "GRAN SASSO, ITALY": "GSSI",
    "GS": "GSSI",
    "MODANE": "MODANE",
    "MODANE, FRANCE": "MODANE",
    "SOUDAN": "SOUDAN",
    "SOUDAN, USA": "SOUDAN",
    "SURF": "SOUDAN",
    "STAWELL": "SUPL",
    "STAWELL, AUSTRALIA": "SUPL",
    "SUPL": "SUPL",
    "CAPE TOWN": "CAPETOWN",
    "CAPE TOWN, AFRICA": "CAPETOWN",
    "PAUL": "CAPETOWN",
}

def get_site_thetaiso_loc(site):
    """Return location tuple expected by ThetaIso: (lat_rad, lon_rad, depth_km)."""
    lat_deg, lon_deg, height_m = SITE_COORDS[normalize_site_key(site)]
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    depth_km = -height_m / 1000.0
    return (lat_rad, lon_rad, depth_km)

def get_site_location(site):
    lat, lon, height = SITE_COORDS[normalize_site_key(site)]
    return EarthLocation.from_geodetic(
        lon=lon * u.deg,
        lat=lat * u.deg,
        height=height * u.m,
    )

sites = {
    key: {"loc": get_site_thetaiso_loc(key)}
    for key in SITE_COORDS
}


# ---------------------------------------------------------------------------
# Solar-beam SRDM angle helpers (Step D2)
# These compute the file-facing SRDM beam angle for SRDMBeam flux lookup.
# Do NOT use ThetaIso() for this purpose — that is the galactic halo-wind angle.
# Convention: 0 deg = overhead, 90 deg = horizon, 180 deg = nadir.
# ---------------------------------------------------------------------------

def _normalize_location_to_earthlocation(location):
    """Convert site string or EarthLocation to EarthLocation."""
    if isinstance(location, EarthLocation):
        return location
    if isinstance(location, str):
        return get_site_location(location)
    raise TypeError(f"Expected site string or EarthLocation, got {type(location)}")


def _srdm_iso_from_alt_deg(alt_deg):
    """Return srdm_isoangle_deg from solar altitude in degrees."""
    return float(np.clip(90.0 - alt_deg, 0.0, 180.0))


def _internal_gamma_from_alt_deg(alt_deg):
    """Return propagation-code internal gamma from solar altitude in degrees."""
    return float(np.clip(90.0 + alt_deg, 0.0, 180.0))


def _solar_alt_deg(location, time):
    """Return solar altitude in degrees using Astropy AltAz frame."""
    earth_loc = _normalize_location_to_earthlocation(location)
    altaz_frame = AltAz(obstime=time, location=earth_loc)
    sun = get_sun(time)
    sun_altaz = sun.transform_to(altaz_frame)
    return float(sun_altaz.alt.to_value(u.deg))


def SolarBeamIsoAngle(location, time):
    """File-facing SRDM beam angle in degrees for SRDMBeam flux lookup.

    Returns srdm_isoangle_deg: 0=overhead, 90=horizon, 180=nadir.
    location: site string or EarthLocation.
    time: astropy.time.Time.
    """
    alt_deg = _solar_alt_deg(location, time)
    return _srdm_iso_from_alt_deg(alt_deg)


def SolarBeamInternalGamma(location, time):
    """Propagation-code internal gamma in degrees (diagnostic only).

    Returns gamma_internal_deg: 0=nadir, 90=horizon, 180=overhead.
    Not used for ring-index lookup; use SolarBeamIsoAngle() for that.
    location: site string or EarthLocation.
    time: astropy.time.Time.
    """
    alt_deg = _solar_alt_deg(location, time)
    return _internal_gamma_from_alt_deg(alt_deg)


def angle_to_ring_index(srdm_isoangle_deg, ring_count):
    """Map srdm_isoangle_deg to a zero-based integer ring index.

    ring_index = floor(srdm_isoangle_deg / 180.0 * ring_count)
    Clipped to [0, ring_count - 1]. Out-of-range angles are clipped.
    ring_count must be a positive integer.
    """
    if ring_count <= 0:
        raise ValueError(f"ring_count must be positive, got {ring_count}")
    raw = int(np.floor(srdm_isoangle_deg / 180.0 * ring_count))
    return min(max(raw, 0), ring_count - 1)


def get_solar_angle_limits(location, date):
    """Return (min_srdm_isoangle_deg, max_srdm_isoangle_deg) over one sidereal day.

    date: [day, month, year] list, e.g. [8, 8, 2024].
    location: site string or EarthLocation.
    Samples at 1-minute cadence.
    """
    _, angles = solar_daily_angle_series(location, date, cadence_minutes=1)
    return float(np.min(angles)), float(np.max(angles))


def solar_daily_angle_series(location, date, cadence_minutes=10):
    """Return (times, srdm_isoangle_deg_array) sampled over one UTC day.

    date: [day, month, year] list, e.g. [8, 8, 2024].
    location: site string or EarthLocation.
    Returns (astropy.time.Time array, numpy.ndarray of float64).
    """
    d, m, y = int(date[0]), int(date[1]), int(date[2])
    t0 = Time(f"{y:04d}-{m:02d}-{d:02d}T00:00:00", format="isot", scale="utc")
    n_steps = int(24 * 60 / cadence_minutes)
    offsets = np.arange(n_steps) * cadence_minutes / (24.0 * 60.0)
    times = t0 + offsets * u.day

    earth_loc = _normalize_location_to_earthlocation(location)
    altaz_frame = AltAz(obstime=times, location=earth_loc)
    sun = get_sun(times)
    sun_altaz = sun.transform_to(altaz_frame)
    alt_degs = sun_altaz.alt.to_value(u.deg)
    angles = np.clip(90.0 - alt_degs, 0.0, 180.0)
    return times, angles
