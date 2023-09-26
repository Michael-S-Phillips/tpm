import numpy as np
import scipy.optimize as optimize
from scipy.integrate import quad
from IPython import display
import matplotlib
import matplotlib.pylab as plt
matplotlib.use('TkAgg')
import time
import tpm_utils
tpm_utils = tpm_utils.tpm_utils()

def TPM(di, li, flag, cap, tol, tlim):
    ''' 
    This function solves the 1-D heat diffusion equation for a given latitude
    of an airless planetary body. The radius range is set up for Mercury, but 
    can be edited to suite any airless body. 

    Inputs:
        di -    This is the number of depth steps to take per skin depth. Skin
                depth is calculated from the diffusivity, alpha, which is calculated
                initially based on the conductivity, kT, the density, rho0, and the
                diurnal time period, tao.
        li -    This is the latitude. Latitude ranges from 0 to 89, so li - 1 is
                the latitude that will be used.
        flag -  The flag is either 'hot' or 'warm' to indicate which vector to
                use for the radius as a function of time. Mercury has a high
                eccentricity, so it's useful to solve for the hot and warm poles. 
        cap -  'regolith' to use a regolith depth-density profile, use
                'basalt' to use a constant basaltic density profile.
        tol -   This is the tolerance threshold below which the function is
                said to converge. A reasonable choice is 5e-3, or 1e-3.
        tlim -  This is the minimum amount of time to rotate expressed in
                Mercury days. Sometimes, the function will converge very quickly, so
                use 1.5 or 2 if you want to program to run at least 1.5 or 2 Merucry
                days. 

    Outputs:
        h -     The depth step at the surface
        x -     A vector containing the depths in meters. Plot against Tl.
        l -     Vector containing the time expressed in Mercury days, plot
                against Tsl.
        dt -    This is the time step expressed in seconds. 
        t -     This is the final time expressed in seconds.
        Tl -    The temperature depth profiles for 0,90,180,270 longitudes.
        Tsl -   Surface temperatures saved for every tenth time step.
        tti -   Surface thermal inertia at each time step.
    '''

    # Constants
    Ab = 0.06  # Albedo coefficient
    emis = 0.95  # Emissivity coefficient
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant
    invab = 1 - Ab  # Inverse albedo coefficient
    lat = np.linspace(0, 89, 90)  # Latitudes array

    # Define depth-dependent density function
    if cap == 'regolith':
        rho = lambda z: 3100 * ((z + 0.122) / (z + 0.18))
    elif cap == 'basalt':
        rho = lambda z: 2840 + z * 10

    # Thermal conductivity coefficients
    l0 = 75e-6  # Coefficient for initial depth
    l1 = l0 / 100  # Coefficient for final depth
    Beta0 = 4 * sigma * emis * l0  # Initial thermal conductivity coefficient
    Beta1 = 4 * sigma * emis * l1  # Final thermal conductivity coefficient
    kcT = 4e-3  # Thermal conductivity coefficient for top layer
    kcB = 7e-3  # Thermal conductivity coefficient for bottom layer

    def kT(T):
        return kcT + Beta0 * T**3  # Thermal conductivity function

    # Confidence interval values for heat capacity coefficients
    CI = np.array([[-147.190898324622, 4.11951434554240, -0.00529042037670872, 2.42723186129553e-06, -4.75097411647327e-10],
                [-80.5149, 5.1360, -0.0087, 7.6388e-6, -2.3769e-9]])

    def cp(T):
        return CI[1, 0] + CI[1, 1] * T + CI[1, 2] * T**2 + CI[1, 3] * T**3 + CI[1, 4] * T**4  # Heat capacity function

    # Calculate thermal inertia parameters
    CIrho, _ = quad(rho, 0, 10)  # Density integral
    alpha0 = kT(700) / (CIrho * cp(700))  # Thermal inertia parameter
    tao = 175.94  # Period of a solar day in Earth days
    dw = tpm_utils.skindepth(alpha0, tao)  # Skin depth
    dy = dw / di  # Grid spacing

    # Initialize some variables
    denergy = 0  # Energy change
    dflux = 0  # Flux change
    Tsum = 0  # Sum of temperatures
    dtemp = 1  # Temperature change
    ind = 0  # Index counter

    # Set up grid parameters
    L = dw / 10  # Length of top layer
    n = int(np.ceil(L / dy))  # Number of grid points in top layer
    h = L / n  # Grid spacing in top layer

    dyy = dy * 2  # Grid spacing constant
    mL = 1.5 - L  # Length of middle layer
    mn = int(np.ceil(mL / dyy))  # Number of grid points in middle layer
    mh = mL / mn  # Grid spacing in middle layer

    bb = 10  # Length of bottom layer
    dy3 = dyy * 5  # Grid spacing constant
    LL = bb - mL - L  # Length of bottom two layers
    nn = int(np.ceil(LL / dy3))  # Number of grid points in bottom two layers
    hh = LL / nn  # Grid spacing in bottom two layers

    N = n + mn + nn  # Total number of grid points

    x = np.concatenate([np.linspace(0, L, n + 1),
                        np.linspace(L + 0.5 * h, L + mL, mn + 1),
                        np.linspace(L + mL, L + mL + LL + 0.5 * hh, nn)])
    x[0] = 1e-4  # Set the first element of x as 1e-4

    # thermal conductivity (no radiation)
    def ki(x, bb):
        return kcB - ((kcB - kcT) * ((rho(bb) - rho(x)) / (rho(bb) - rho(0))))

    # radiative thermal conductivity empirical formula
    def kc(ki, Beta, T):
        return ki + Beta * T**3

    # Define depth-dependent density function
    def rho(z):
        if cap == 'regolith':
            return 3100 * ((z + 0.122) / (z + 0.18))
        elif cap == 'basalt':
            return 2840 + z * 10

    # Define some constants and parameters
    dz = np.zeros(N + 2)
    dz[0:n] = h
    dz[n:mn] = mh
    dz[mn:] = hh

    d3z = np.ones(N + 2)
    d3z[1:] = dz[1:] * dz[:-1] * (dz[1:] + dz[:-1])
    d3z[0] = d3z[1]

    zz = np.zeros(N + 2)
    for i in range(N + 2):
        if i == 0:
            zz[i] = dz[i]
        else:
            zz[i] = zz[i - 1] + dz[i]

    il = l0 * np.exp(-50 * zz)
    Beta = 4 * sigma * emis * il

    alpha = np.zeros(N + 2)
    A = np.zeros((N + 2, N + 2))
    B = np.zeros((N + 2, N + 2))
    const = np.zeros(N + 2)

    p = np.zeros(N + 2)
    q = np.zeros(N + 2)

    for i in range(N + 1):
        p[i] = 2 * (dz[i + 1]) / d3z[i]

    for i in range(1, N + 2):
        q[i] = 2 * (dz[i]) / d3z[i]

    p[-1] = p[-2]
    q[0] = q[1]

    a0 = p[0] * kT(700)
    b0 = q[0] * kT(700)

    # Period of a year in seconds
    Psec = (0.387098**(3/2)) * 365.24 * 24 * 3600
    # Spin-orbit resonance number, e.g., 3:2 would be RN = 3/2, 2:1 would be RN = 2/1, etc.
    RN = 3/2
    # This is the numerator of RN
    sidrotperday = 3
    # Sidereal rotation rate in radians/second
    sid_rot = 2 * np.pi * RN / Psec
    # Sidereal rotation period in seconds
    Psid = Psec / RN
    # Period of a solar day
    taos_inv = abs(1 / Psid - 1 / Psec)
    # Calculate the solar day period
    taos = 1 / taos_inv
    print(f'The period of a day is {taos} s')

    dt = 60 * (rho(h) * cp(700) / (2 * (a0 + b0)))
    print(f'The time step is {dt} s')

    tstep = int(np.ceil(taos / dt))  # Number of time steps
    lon = np.linspace(0, 359, tstep)  # Longitudes array
    a_lon = np.linspace(0, 2 * np.pi, tstep)  # apparent longitudes array


    # Calculate orbital parameters for Mercury
    if flag == 'hot':
        pd = 0.3075  # Perihelion distance in AU
        e = 0.206  # Eccentricity
        af = 0
        d = np.linspace(0, taos, tstep) / (3600 * 24)

        R = np.zeros(tstep)
        tanom = np.zeros(tstep)

        for i in range(tstep):
            R[i], tanom[i] = tpm_utils.kepler4(d[i], pd, e)

    elif flag == 'warm':
        pd = 0.3075  # Perihelion distance in AU
        e = 0.206  # Eccentricity
        af = 270 / 360  # The numerator is the longitude in °E
        t1 = af * taos
        t2 = (1 + af) * taos
        d = np.linspace(t1, t2, tstep) / (3600 * 24)

        R = np.zeros(tstep)
        tanom = np.zeros(tstep)

        for i in range(tstep):
            R[i], tanom[i] = tpm_utils.kepler4(d[i], pd, e)

    else:
        print('Set flag to either hot or warm')

    if len(R) < tstep:
        R = np.append(R, R[-1])
    elif len(R) > tstep:
        R = R[:-1]

    # orb_rot_rate = (np.roll(tanom, -1) - tanom) / dt
    # orb_rot_rate[orb_rot_rate < 0] = orb_rot_rate[(np.arange(len(orb_rot_rate)) + 1) % len(orb_rot_rate)]

    # diff_rot = (sid_rot - orb_rot_rate) * dt

    # Calculate orb_rot_rate
    orb_rot_rate = (np.roll(tanom, -1) - tanom) / dt

    # Replace negative values in orb_rot_rate
    for i in range(len(orb_rot_rate)):
        if orb_rot_rate[i] < 0:
            orb_rot_rate[i] = orb_rot_rate[(i + 1) % len(orb_rot_rate)]

    # Calculate diff_rot
    diff_rot = (sid_rot - orb_rot_rate) * dt

    for i in range(1, tstep):
        a_lon[i] = a_lon[i - 1] + diff_rot[i - 1]
    a_lon = np.rad2deg(a_lon)

    # Define thermal inertia function
    def ti(T):
        return np.sqrt(kT(T) * rho(h) * cp(T))

    if li < 11:
        theta = ti(350) * np.sqrt(2 * np.pi / taos) / (emis * sigma * (tpm_utils.Tss(invab, sigma, emis, R)**3))
        if theta > 4:
            ftss = 0.75
        else:
            ftss = np.log10(theta) * 0.065 + 0.65
    else:
        ftss = 1.75

    # Get initial surface temperatures
    Ts, Qs, Tdeep = tpm_utils.sol_flux(lat, a_lon, invab, sigma, emis, R, ftss)
    print(Qs.shape)

    # Set the initial temperature distribution
    i75 = np.where(lat == 75)[0][0]
    Tsurf = Ts[li, 0]
    m = -60.9302
    b = 196.2792
    Tdepth = np.cos(np.deg2rad(li - 1)) * m * np.log10(kcT) + b
    if flag == 'warm':
        Tdepth = Tdepth - 70

    T = np.zeros(len(x))
    T[0] = Tsurf
    T[1:] = Tdepth

    # If you only want to save specific temperature profiles, use this:
    Tl = np.zeros((len(x), 4))

    # If you want to save all temperature profiles, use this:
    sf = 1
    Tsl = np.zeros((1, int(tstep / sf)))
    l = np.linspace(0 + af, sidrotperday + af, Tsl.shape[1])

    # Making a vector to save every other temperature profile
    Tl_all = np.zeros((len(x), int(tstep / sf)))

    # Enter the time advance loop
    tkern = 0
    mtime = 0
    max_surf_index = -1
    dT = np.ones(tstep)
    tti = np.ones(tstep)

    # set up a figure to plot the temperature profiles
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('log10(x)')
    ax.set_ylabel('T (k)')
    ax.set_title('Temperature vs. Depth')
    ax.set_ylim([0, 850])
    line, = ax.semilogx(x, np.zeros(len(x)))

    z = np.zeros(N + 2)
    alpha = np.zeros(N + 2)

    Dm = np.zeros(N + 2)
    Dp = np.zeros(N + 2)
    
    const = np.zeros(N + 2)

    denergy = 0
    dflux = 0
    Tsum = 0
    dt_guess = 0

    while (tkern < tlim * tstep) or (abs(dtemp) > tol and np.min(dT) > tol):

        # update the model time
        mtime = 1 + tkern % tstep
        t = tkern * dt

        for i in range(1, N + 3):
            if i == 1:
                z[i - 1] = dz[i - 1]
            else:
                z[i - 1] = z[i - 2] + dz[i - 1]

            alpha[i - 1] = kc(ki(z[i - 1], bb), Beta[i - 1], T[i - 1]) / (rho(z[i - 1]) * cp(T[i - 1]))

        # Load the diffusion coefficient array (make it a column vector)
        D = alpha * np.ones(N + 2)
        
        # Load Dm with average values D(j-1/2) and Dp with D(j+1/2)
        Dm[1:N + 1] = 0.5 * (D[1:N + 1] + D[0:N])
        Dp[1:N + 1] = 0.5 * (D[1:N + 1] + D[2:N + 2])

        # Load A and B at interior points
        for i in range(1, N + 3):
            if i < mn + 1:
                const[i - 1] = 2 * (h**2) / dt
            elif mn <= i < nn:
                const[i - 1] = 2 * (mh**2) / dt
            else:
                const[i - 1] = 2 * (hh**2) / dt

        # Create the matrices A and B by loading them with zeros
        A = np.zeros((N + 2, N + 2))
        B = np.zeros((N + 2, N + 2))

        for j in range(1, N + 1):
            A[j, j - 1] = -Dm[j]
            A[j, j] = const[j] + (Dm[j] + Dp[j])
            A[j, j + 1] = -Dp[j]
            B[j, j - 1] = Dm[j]
            B[j, j] = const[j] - (Dm[j] + Dp[j])
            B[j, j + 1] = Dp[j]

        # Load the boundary conditions into A and B
        A[0, 0] = 0.5
        A[0, 1] = 0.5
        B[0, 0] = 0.

        A[N + 1, N] = 0.5
        A[N + 1, N + 1] = 0.5
        B[N + 1, N + 1] = 0.
        
        # Find the right-hand side for the solution at interior points
        r = np.dot(B, T)
        
        # Apply the boundary conditions
        tempgrad = (kc(ki(z[0], bb), Beta[0], T[0]) / dz[0]) * (T[0] - T[1])
        radiated = sigma * emis * (T[0] ** 4)
        
        T1 = T[1]
        fenergy = lambda T0: Qs[li, mtime-1] - sigma * emis * (T0 ** 4) - (kc(ki(z[0], bb), Beta[0], T0) / dz[0]) * (T0 - T1)
        
        # balance the energy equation using Newton's method
        r[0] = optimize.root_scalar(fenergy, bracket=[0, 1000]).root
        
        if r[0] < 0:
            r[0] = 0
        r[-1] = T[-2]  # or some other value

        # Do the linear solve to update T
        T = np.linalg.solve(A, r)
        
        denergy += Qs[li, mtime-1] - radiated - tempgrad
        dflux += -4 * emis * sigma * (T[0] ** 3)
        dtemp = kT(T[0]) / (denergy / dflux)  # denergy / dflux / kT(T[0])
        Tsum += T[0]
        # avgT = Tsum / mtime
        avgT = Tsum / tkern
        dT[mtime-1] = abs(avgT - np.mean(T))
        tti[mtime-1] = ti(T[0])

        # Make a plot of T every once in a while
        # Tmax = 1000
        # Tmin = 0
        if mtime % sf == 0:
            Tsl[0, (mtime-1) // sf] = T[0]
            line.set_ydata(T)
            ax.set_title(f'Temperature vs. Depth: Time {np.round(t/taos, 2)} Mercury Days\n tkern: {tkern}, {tlim * tstep}\ncondition 1: {(tkern < tlim * tstep)}, condition 2: {(abs(dtemp) > tol and np.min(dT) > tol)}\nmin dT: {np.min(dT)}, tol: {tol}, r[0]: {np.round(r[0], 2)}\n dtemp: {dtemp}, denergy: {denergy}\nkT: {kT(T[0])}, dflux: {dflux}\n tempgrad: {tempgrad}, radiated: {radiated}')
            # ax.set_title(f'Temperature vs. Depth: Time {np.round(tkern/tstep, 2)} Mercury Days\n min dT: {np.min(dT)}, tol: {tol}\n dtemp: {dtemp}, r0: {r[0]}')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.00001)
            max_surf_index = np.argmax(Tsl)
            Tl_all[:, (mtime-1) // sf] = T

        # Display some values every once in a while
        if mtime % (tstep // 4) == 0:
            print(f"{round(tkern / tstep, 2)} rotations complete")
            print(f"surf T is {T[0]}")
            print(f"dtemp is {dtemp}")
            print(f"dT is {np.min(dT)}")
            # fig, ax = plt.subplots()
            # ax.set_xlabel('log10(x)')
            # ax.set_ylabel('T (k)')
            # ax.set_title('Temperature vs. Depth')
            # ax.set_ylim([0, 1000])
            # line, = ax.semilogx(x, np.zeros(len(x)))
            Tl[:, ind % 4] = T
            ind += 1
            if ind == 4:
                Tl_1 = Tl
            if ind == tlim // 2:
                Tl_2 = Tl

        # Update Tdepth after 1 rotation, reset flux values
        if mtime % tstep == 0:
            tdguess = np.mean(Tsl)
            T[-nn:] = tdguess
            denergy = 0
            dflux = 0

        # Check progress every 30 rotations
        # if tkern % (30 * tstep) == 0:
        #     print('30')

        if mtime-1 == max_surf_index:
            Tl_max = T
        
        tkern += 1

    # toc()
    mdT = np.min(dT)
    print(f"final dT is {mdT}")
    print('converged!')

    return h, mh, hh, x, l, lon, dt, t, Tl, Tl_all, Tsl, tti, dw, Tl_max, tanom

