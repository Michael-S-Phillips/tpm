import numpy as np
import math

class tpm_utils:

    def __init__(self):
        pass

    def kepler4(self, t, q, ecc):
        kgauss = 0.01720209895
        e2 = 0
        fac = 0.5 * ecc
        i = 0
        tau = kgauss * t
        
        while True:
            i += 1
            e20 = e2
            a = 1.5 * np.sqrt(fac / (q ** 3)) * tau
            b = (np.sqrt(a ** 2 + 1) + a) ** (1 / 3)
            u = b - 1 / b
            u2 = u ** 2
            e2 = u2 * (1 - ecc) / fac
            c1, c2, c3 = self.stumpff(e2)
            fac = 3 * ecc * c3
            
            # Check for convergence
            if (np.abs(e2 - e20) < 1e-9) or (i > 15):
                break
        
        if i > 15:
            print('\n\n   more than 15 iterations in kepler4 \n\n')
            # Handle the case of more than 15 iterations as needed
        
        # Heliocentric distance (AU)
        r = q * (1 + u2 * c2 * ecc / fac)
        # x and y coordinates
        x = q * (1 - u2 * c2 / fac)
        y = q * np.sqrt((1 + ecc) / fac) * u * c1
        # True anomaly (radians)
        tanom = np.arctan2(y, x)
        
        return r, tanom

    def stumpff(self, e2):
        c1, c2, c3 = 0, 0, 0
        deltac = 1
        n = 1
        
        while True:
            c1 += deltac
            deltac /= (2 * n)
            c2 += deltac
            deltac /= (2 * n + 1)
            c3 += deltac
            deltac *= -e2
            n += 1
            
            # Check for convergence
            if np.abs(deltac) < 1e-12:
                break
        
        return c1, c2, c3

    def skindepth(self, alpha, tao):
        """
        Solve for skin depth given a diffusivity and a period in days.

        Parameters:
        - alpha (float): Diffusivity in m^2/s.
        - tao (float): Period in days.

        Returns:
        - dw (float): Skin depth.
        """
        taos = tao * 24 * 3600  # Period in seconds
        w = 2 * np.pi / taos  # Circular frequency in seconds
        dw = np.sqrt(2 * alpha / w)  # Skin depth
        return dw


    def sol_flux(self, lat, lon, invab, sigma, emis, R, ftss):
        """
        Calculate surface temperatures for given latitudes, longitudes, surface
        reflectance, Stefan-Boltzmann constant, surface emissivity, and distance
        from the Sun in AU.

        Parameters:
        - lat (numpy.ndarray): Latitude values in radians.
        - lon (numpy.ndarray): Longitude values in radians.
        - invab (float): Solar flux modifier.
        - sigma (float): Stefan-Boltzmann constant.
        - emis (float): Surface emissivity.
        - R (numpy.ndarray): Distance from the Sun in AU.
        - ftss (float): Factor for subsolar temperature.

        Returns:
        - Ts (numpy.ndarray): Surface temperatures.
        - Qs (numpy.ndarray): Solar flux.
        - Tdeep (numpy.ndarray): Deep temperatures.
        """
        dtor = np.pi / 180
        latsz, lonsz = lat.shape[0], lon.shape[0]
        # print(f'lat size: {latsz}, lon size: {lonsz}')
        Fsun = 1367 / (R ** 2)  # Solar flux at radius R from the Sun
        Tss = ((Fsun[0] * invab / (sigma * emis)) ** 0.25)  # Subsolar temperature estimate
        Qs = np.zeros((latsz, lonsz))
        Ts = np.zeros((latsz, lonsz))
        Tdeep = np.zeros((latsz, lonsz))
        solang = np.zeros((latsz, lonsz))

        for i in range(latsz):
            for j in range(lonsz):
                solang[i, j] = self.sunang(dtor * lat[i], dtor * lon[j])
                Qs[i, j] = solang[i, j] * Fsun[j] * invab  # Subsolar temperature given solar flux, Fsun, and a lat, lon
                Ts[i, j] = solang[i, j] * Tss
                Tdeep[i, j] = ftss * Ts[i, j]

                # For areas in permanent shadow
                if Ts[i, j] <= 0:
                    Ts[i, j] = 13.7

        return Ts, Qs, Tdeep

    def sph_area(self, latmin, latmax, lonmin, lonmax, totarea):
        latfrac = (math.cos(latmin) - math.cos(latmax)) / (2 * math.pi)
        lonfrac = (math.cos(lonmin) - math.cos(lonmax)) / (2 * math.pi)
        
        pa = latfrac * lonfrac
        SA = pa * totarea
        
        return SA

    def sunang(self, lat, lon):
        sublat = 0
        sublon = 0

        solang = math.sin(sublat) * math.sin(lat) + math.cos(sublat) * math.cos(lat) * math.cos(abs(lon - sublon))
        
        if solang < 0:
            solang = 0
        elif 0 < solang < 0.0261799:
            solang = solang * (-math.cos(math.pi * solang / 0.0261799) / 2 + 0.5)
        
        return solang

    def Tss(self, invab, sigma, emis, R):
        Fsun = 1367 / (R[0] ** 2)  # Solar flux at radius R from the sun
        Tss = ((Fsun * invab / (sigma * emis)) ** 0.25)
        return Tss