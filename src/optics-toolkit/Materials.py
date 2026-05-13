# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .Constants import *
from .Dispersion import *

class Material(ABC): 
    @property
    @abstractmethod
    def MinimumWavelengthOfDispersion(self) -> float: 
        pass

    @property
    @abstractmethod
    def MaximumWavelengthOfDispersion(self) -> float: 
        pass

    @abstractmethod
    def IsWavelengthWithinRange(self, wavelength: float) -> bool: 
        pass

    @property
    @abstractmethod
    def RefractiveIndex(self) -> DispersionCurve: 
        pass

    @property
    @abstractmethod
    def RelativePermittivity(self) -> DispersionCurve: 
        pass

    @property
    @abstractmethod
    def Permittivity(self) -> DispersionCurve: 
        pass

    @property
    @abstractmethod
    def RelativePermeability(self) -> DispersionCurve: 
        pass

    @property
    @abstractmethod
    def Permeability(self) -> DispersionCurve: 
        pass

    @property
    @abstractmethod
    def Impedance(self) -> DispersionCurve: 
        pass

class Material_n(Material): 
    def __init__(
            self, 
            refractiveIndex: DispersionCurve
    ): 
        if not isinstance(refractiveIndex, DispersionCurve): 
            raise TypeError('The dispersion characteristics of an instance of the Material_n class must be given through the refractive index in the form of a class derived from DispersionCurve.')
        self._refractiveIndex = refractiveIndex
    
    @property
    def RefractiveIndex(self): 
        return self._refractiveIndex
    @RefractiveIndex.setter
    def RefractiveIndex(self, value: DispersionCurve): 
        if not isinstance(value, DispersionCurve): 
            raise TypeError('The dispersion characteristics of an instance of the Material_n class must be given through the refractive index in the form of a class derived from DispersionCurve.')
        self._refractiveIndex = value
    
    @property
    def MinimumWavelengthOfDispersion(self): 
        return self.RefractiveIndex.MinimumWavelength
    
    @property
    def MaximumWavelengthOfDispersion(self): 
        return self.RefractiveIndex.MaximumWavelength
    
    @property
    def RelativePermittivity(self): 
        def epsilon_r_dispersion(wavelength): 
            return np.power(self.RefractiveIndex.GetValue(wavelength), 2.0, dtype=complex)
        return CustomDispersion(epsilon_r_dispersion, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    @property
    def Permittivity(self): 
        def epsilon_dispersion(wavelength): 
            return Constants.EPSILON0 * self.RelativePermittivity.GetValue(wavelength)
        return CustomDispersion(epsilon_dispersion, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    @property
    def RelativePermeability(self): 
        def mu_r_dispersion(wavelength): 
            return complex(1.0)
        return CustomDispersion(mu_r_dispersion, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    @property
    def Permeability(self): 
        def mu_dispersion(wavelength): 
            return complex(Constants.MU0)
        return CustomDispersion(mu_dispersion, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    @property
    def Impedance(self): 
        def eta_dispersion(wavelength): 
            return np.sqrt(self.Permeability.GetValue(wavelength) / self.Permittivity.GetValue(wavelength), dtype=complex)
        return CustomDispersion(eta_dispersion, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    def IsWavelengthWithinRange(self, wavelength: float) -> bool: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength >= self.MinimumWavelengthOfDispersion and wavelength <= self.MaximumWavelengthOfDispersion: 
            return True
        else: 
            return False
    
class Material_em(Material): 
    def __init__(
            self, 
            relativePermittivity: DispersionCurve, 
            relativePermeability: DispersionCurve
    ): 
        if not isinstance(relativePermittivity, DispersionCurve): 
            raise TypeError('The dispersion characteristics of the relative permittivity of the material must be delivered as an instance of a class derived from DispersionCurve.')
        if not isinstance(relativePermeability, DispersionCurve): 
            raise TypeError('The dispersion characteristics of the relative permeability of the material must be delivered as an instance of a class derived from DispersionCurve.')
        self._relativePermittivity = relativePermittivity
        self._relativePermeability = relativePermeability
    
    @property
    def RelativePermittivity(self): 
        return self._relativePermittivity
    @RelativePermittivity.setter
    def RelativePermittivity(self, value): 
        if not isinstance(value, DispersionCurve): 
            raise TypeError('The dispersion characteristics of the relative permittivity of the material must be delivered as an instance of a class derived from DispersionCurve.')
        self._relativePermittivity = value
    
    @property
    def RelativePermeability(self): 
        return self._relativePermeability
    @RelativePermeability.setter
    def RelativePermeability(self, value): 
        if not isinstance(value, DispersionCurve): 
            raise TypeError('The dispersion characteristics of the relative permittivity of the material must be delivered as an instance of a class derived from DispersionCurve.')
        self._relativePermeability = value
    
    @property
    def MinimumWavelengthOfDispersion(self): 
        return np.nanmin((self.RelativePermittivity.MinimumWavelength, self.RelativePermeability.MinimumWavelength))
    
    @property
    def MaximumWavelengthOfDispersion(self): 
        return np.nanmax((self.RelativePermittivity.MaximumWavelength, self.RelativePermeability.MaximumWavelength))
    
    def IsWavelengthWithinRange(self, wavelength: float) -> bool: 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength >= self.MinimumWavelengthOfDispersion and wavelength <= self.MinimumWavelengthOfDispersion: 
            return True
        else: 
            return False
    
    @property
    def RefractiveIndex(self): 
        def n_dispersion(wavelength): 
            n = np.sqrt(self.RelativePermittivity.GetValue(wavelength) * self.RelativePermeability.GetValue(wavelength), dtype=complex)
            return n
        return CustomDispersion(n_dispersion, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)

    @property
    def Permittivity(self): 
        def epsilon(wavelength): 
            return Constants.EPSILON0 * self.RelativePermittivity.GetValue(wavelength)
        return CustomDispersion(epsilon, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    @property
    def Permeability(self): 
        def mu(wavelength): 
            return Constants.MU0 * self.RelativePermeability.GetValue(wavelength)
        return CustomDispersion(mu, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)
    
    @property
    def Impedance(self): 
        def eta(wavelength): 
            return np.sqrt(self.Permeability.GetValue(wavelength)/self.Permittivity.GetValue(wavelength), dtype=complex)
        return CustomDispersion(eta, self.MinimumWavelengthOfDispersion, self.MaximumWavelengthOfDispersion)