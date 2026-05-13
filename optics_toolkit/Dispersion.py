# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
from abc import ABC, abstractmethod

from customlib.Constants import Constants

class DispersionCurve(ABC): 
    @property
    @abstractmethod
    def MinimumWavelength(self) -> float: 
        pass

    @MinimumWavelength.setter
    @abstractmethod
    def MinimumWavelength(self, value): 
        pass

    @property
    @abstractmethod
    def MaximumWavelength(self) -> float: 
        pass

    @MaximumWavelength.setter
    @abstractmethod
    def MaximumWavelength(self, value): 
        pass

    @property
    @abstractmethod
    def DispersionFormula(self) -> callable:
        pass

    @abstractmethod
    def GetValue(self, wavelength: float) -> complex: 
        pass

    @abstractmethod
    def GetRealValue(self, wavelength: float) -> float: 
        pass

    @abstractmethod
    def GetImaginaryPart(self, wavelength: float) -> float: 
        pass

class NonDispersive(DispersionCurve): 
    def __init__(
            self,  
            constantValue: complex = 1.0+0.0j, 
            minimumWavelength: float = 0.0, 
            maximumWavelength: float = 1e-3
    ): 
        if not isinstance(minimumWavelength, float): 
            raise TypeError('The minimum wavelength of the range in which the dispersion curve is defined must be a float.')
        if minimumWavelength < 0.0: 
            raise ValueError('The minimum wavelength of the range in which the dispersion curve is defined must be positive.')
        if not isinstance(maximumWavelength, float): 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be a float.')
        if maximumWavelength < minimumWavelength: 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be larger than the minimum wavelength of the range.')
        if not (isinstance(constantValue, int) or isinstance(constantValue, float) or isinstance(constantValue, complex)): 
            raise TypeError('The constant value of the dispersive function must be given as an int, float or complex.')
        
        self._minimumWavelength = minimumWavelength
        self._maximumWavelength = maximumWavelength
        self._constantValue = complex(constantValue)
    
    @property
    def MinimumWavelength(self): 
        return self._minimumWavelength
    @MinimumWavelength.setter
    def MinimumWavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The minimum wavelength of the range in which the dispersion curve is defined must be a float.')
        if value < 0.0: 
            raise ValueError('The minimum wavelength of the range in which the dispersion curve is defined must be positive.')
        self._minimumWavelength = value
    
    @property
    def MaximumWavelength(self): 
        return self._maximumWavelength
    @MaximumWavelength.setter
    def MaximumWavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be a float.')
        if value < self.MinimumWavelength: 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be larger than the minimum wavelength of the range.')
        self._maximumWavelength = value
    
    @property
    def DispersionFormula(self, wavelength): 
        def dispersionF(wavelength): 
            return self.ConstantValue
        return dispersionF
    
    @property
    def ConstantValue(self): 
        return self._constantValue
    @ConstantValue.setter
    def ConstantValue(self, value): 
        if not (isinstance(value, int) or isinstance(value, float) or isinstance(value, complex)): 
            raise TypeError('The constant value of the dispersive function must be given as an int, float or complex.')
        self._constantValue = complex(value)
    
    def GetValue(self, wavelength: float) -> complex: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength < self.MinimumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        if wavelength > self.MaximumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        return self.ConstantValue
    
    def GetRealValue(self, wavelength: float) -> float: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength < self.MinimumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        if wavelength > self.MaximumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        return np.real(self.ConstantValue)
    
    def GetImaginaryPart(self, wavelength: float) -> float: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength < self.MinimumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        if wavelength > self.MaximumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        return np.imag(self.ConstantValue)

class CustomDispersion(DispersionCurve): 
    def __init__(
            self, 
            dispersionFormula: callable,
            minimumWavelength: float = 0.0, 
            maximumWavelength: float = 1e-3
    ): 
        if not isinstance(minimumWavelength, float): 
            raise TypeError('The minimum wavelength of the range in which the dispersion curve is defined must be a float.')
        if minimumWavelength < 0.0: 
            raise ValueError('The minimum wavelength of the range in which the dispersion curve is defined must be positive.')
        if not isinstance(maximumWavelength, float): 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be a float.')
        if maximumWavelength < minimumWavelength: 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be larger than the minimum wavelength of the range.')
        
        self._minimumWavelength = minimumWavelength
        self._maximumWavelength = maximumWavelength
        self._dispersionFormula = dispersionFormula

    @property
    def MinimumWavelength(self): 
        return self._minimumWavelength
    @MinimumWavelength.setter
    def MinimumWavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The minimum wavelength of the range in which the dispersion curve is defined must be a float.')
        if value < 0.0: 
            raise ValueError('The minimum wavelength of the range in which the dispersion curve is defined must be positive.')
        self._minimumWavelength = value
    
    @property
    def MaximumWavelength(self): 
        return self._maximumWavelength
    @MaximumWavelength.setter
    def MaximumWavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be a float.')
        if value < self.MinimumWavelength: 
            raise TypeError('The maximum wavelength of the range in which the dispersion curve is defined must be larger than the minimum wavelength of the range.')
        self._maximumWavelength = value
    
    @property
    def DispersionFormula(self): 
        return self._dispersionFormula
    @DispersionFormula.setter
    def DispersionFormula(self, value): 
        self._dispersionFormula = value
    
    def GetValue(self, wavelength: float) -> complex: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength < self.MinimumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        if wavelength > self.MaximumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        return complex(self.DispersionFormula(wavelength))
    
    def GetRealValue(self, wavelength: float) -> float: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength < self.MinimumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        if wavelength > self.MaximumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        return np.real(self.GetValue(wavelength))
    
    def GetImaginaryPart(self, wavelength: float) -> float: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength < self.MinimumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        if wavelength > self.MaximumWavelength: 
            raise ValueError('The wavelength must be within the allowed range.')
        return np.imag(self.GetValue(wavelength))
        
class SellmeierDispersion(DispersionCurve):
    def __init__(
            self, 
            n0: float, 
            n1: float, 
            n2: float, 
            n3: float, 
            d1: float, 
            minimumWavelength: float, 
            maximumWavelength: float
    ):
        if not isinstance(n0, float): 
            raise TypeError()
        if not isinstance(n1, float): 
            raise TypeError()
        if not isinstance(n2, float): 
            raise TypeError() 
        if not isinstance(n3, float): 
            raise TypeError()
        if not isinstance(d1, float): 
            raise TypeError()
        if not isinstance(minimumWavelength, float): 
            raise TypeError()
        if minimumWavelength < 0.0: 
            raise ValueError()
        if not isinstance(maximumWavelength, float): 
            raise TypeError()
        if maximumWavelength < minimumWavelength: 
            raise ValueError()
        self._n0 = n0
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._d1 = d1
        self._minimumWavelength = minimumWavelength
        self._maximumWavelength = maximumWavelength

    @property
    def N0(self): 
        return self._n0
    @N0.setter
    def N0(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._n0 = value

    @property
    def N1(self): 
        return self._n1
    @N1.setter
    def N1(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._n1 = value 

    @property
    def N2(self): 
        return self._n2
    @N2.setter
    def N2(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._n2 = value 

    @property
    def N3(self): 
        return self._n3
    @N3.setter
    def N3(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._n3 = value 

    @property
    def D1(self): 
        return self._d1
    @D1.setter
    def D1(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._d1 = value 

    @property
    def MinimumWavelength(self): 
        return self._minimumWavelength
    @MinimumWavelength.setter
    def MinimumWavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        if value <= 0.0: 
            raise ValueError()
        self._minimumWavelength = value
    
    @property
    def MaximumWavelength(self): 
        return self._maximumWavelength
    @MaximumWavelength.setter
    def MaximumWavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._maximumWavelength = value
    
    @property
    def DispersionFormula(self): 
        def _dispersionFormula(wavelength): 
            _n = self.N0 + (self.N1 / (np.power(1e6*wavelength,2) - self.D1)) + self.N2 * np.power(1e6*wavelength, 2) + self.N3 * np.power(1e6*wavelength,4)
            _n = np.sqrt(_n)
            return _n
        return _dispersionFormula
    
    def GetValue(self, wavelength: float): 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if wavelength < self.MinimumWavelength: 
            raise ValueError()
        if wavelength > self.MaximumWavelength: 
            raise ValueError()
        return complex(self.DispersionFormula(wavelength))
    
    def GetRealValue(self, wavelength: float): 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if wavelength < self.MinimumWavelength: 
            raise ValueError()
        if wavelength > self.MaximumWavelength: 
            raise ValueError()
        return np.real(self.GetValue(wavelength))
    
    def GetImaginaryPart(self, wavelength: float): 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if wavelength < self.MinimumWavelength: 
            raise ValueError()
        if wavelength > self.MaximumWavelength: 
            raise ValueError()
        return np.imag(self.GetValue(wavelength))






