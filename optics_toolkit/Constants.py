# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np

class Constants:
    C = 299792458
    EPSILON0 = 8.85418782e-12 #m-3 kg-1 s4 A2
    MU0 = 1.25663706e-6 # m kg s-2 A-2
    BOLTZMANN = 1.380649e-23 # m2 kg s-2 K-1
    PLANCK = 6.62607015e-34 # m2 kg / s

    def ETA0(): 
        return np.sqrt(Constants.MU0/Constants.EPSILON0)

class Defaults: 
    defaultFieldFigureSize = (15, 8)
    defaultAmplitudeColourMap = 'jet'
    defaultPhaseColourMap = 'Greys'
    defaultWavefrontColourMap = 'jet'
    defaultIntensityColourMap = 'jet'
    defaultFont = 'Georgia'
    defaultFontsizeSuptitle = 24
    defaultFontsizeTitle = 20
    defaultFontsizeAxes = 14
    defaultAxisTicks = 12
    defaultColourDisplayAccents = 'white'
    defaultSquareFigureSize=(8,8)
    defaultSphericalSurfaceApertureTruncation = 100e-9
