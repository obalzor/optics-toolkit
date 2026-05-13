# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np

from .Constants import Constants

class Miscellanea:
    @staticmethod
    def FormatLengthUnits(amount: float): 
        unit = 'm'
        number = amount
        if np.abs(amount) < 1e-6:
            number = 1e9*amount
            unit = 'nm'
        elif np.abs(amount) >= 1e-6 and np.abs(amount) < 1e-3: 
            number = 1e6*amount
            unit = 'µm'
        elif np.abs(amount) >= 1e-3 and np.abs(amount) < 1.0: 
            number = 1e3*amount
            unit = 'mm'
        elif np.abs(amount) >= 1.0 and np.abs(amount) < 1.0e3: 
            number = amount
            unit = 'm'
        elif np.abs(amount) >= 1.0e3 and np.abs(amount) < 1.0e6: 
            number = 1e-3*amount
            unit = 'km'
        
        return number, unit
    
    @staticmethod
    def FormatLengthString(amount): 
        if isinstance(amount, float): 
            number, unit = Miscellanea.FormatLengthUnits(amount)
            return f'{number:.2f} {unit}'
        elif isinstance(amount, np.ndarray):
            string = '('
            for entries in amount: 
                string += Miscellanea.FormatLengthString(entries) + ', '
            string += ')'
            return string
        
    @staticmethod
    def GetLabelAndExtentFactor(x: np.ndarray, y: np.ndarray): 
        if not isinstance(x, np.ndarray): 
            raise TypeError('The x coordinates must be delivered in the form of a numpy array.')
        if not isinstance(y, np.ndarray): 
            raise TypeError('The y coordinates must be delivered in the form of a numpy array.')
        if not np.issubdtype(x.dtype, float): 
            raise TypeError('The dtype of the x coordinates must be float.')
        if not np.issubdtype(y.dtype, float): 
            raise TypeError('The dtype of the y coordinates must be float.')
        if np.shape(x) != np.shape(y): 
            raise ValueError('The size of both sets of coordinates must be the same.')
        
        labelX = '$x$ [m]'
        labelY = '$y$ [m]'
        factor = 1.0

        windowx = np.nanmax(x) - np.nanmin(x)
        windowy = np.nanmax(y) - np.nanmin(y)

        if windowx < 2.0e-6 and windowy < 2.0e-6: 
            labelX = '$x$ [nm]'
            labelY = '$y$ [nm]'
            factor = 1e9
        elif windowx >= 2.0e-6 and windowx < 2.0e-3 and windowy >= 2.0e-6 and windowy < 2.0e-3: 
            labelX = '$x$ [µm]'
            labelY = '$y$ [µm]'
            factor = 1e6
        elif windowx >= 2.0e-03 and windowx < 2.0 and windowy >= 2.0e-3 and windowy < 2.0: 
            labelX = '$x$ [mm]'
            labelY = '$y$ [mm]'
            factor = 1e3
        elif windowx > 2.0e3 and windowx < 2e6 and windowy > 2.0e3 and windowy < 2.0e6: 
            labelX = '$x$ [km]'
            labelY = '$y$ [km]'
            factor = 1e-3
        return labelX, labelY, factor
