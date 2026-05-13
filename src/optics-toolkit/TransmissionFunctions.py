# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
import copy

from .FieldContainers import FieldContainer, MeshTransversalField, MeshElectricField, RegularTransversalField, RegularElectricField, Domains
from .FourierTransforms import *
from .Dispersion import DispersionCurve, NonDispersive
from .Materials import Material_n, Material

class IdealParaxialLens:
    def __init__(
            self, 
            focalLength: float
    ): 
        if not isinstance(focalLength, float): 
            raise TypeError(f'Error in {IdealParaxialLens.__name__}.{IdealParaxialLens.__init__.__name__} → The focal length of the lens must be a float.')
        self._focalLength = focalLength
    
    @property
    def FocalLength(self): 
        return self._focalLength
    @FocalLength.setter
    def FocalLength(self, value): 
        if not isinstance(value, float): 
            raise TypeError(f'Error in {IdealParaxialLens.__name__}.{IdealParaxialLens.FocalLength.__name__} → The focal length of the lens must be a float.')
        self._focalLength = value
    
    def PropagateFieldThrough(self, incidentField: FieldContainer):
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()
        if incidentField.Domain != Domains.X:
            raise ValueError()
        r = np.sqrt(np.power(incidentField.X, 2) + np.power(incidentField.Y, 2) + np.power(self.FocalLength, 2), dtype=float)
        lensFunction = -np.sign(self.FocalLength) * np.real(incidentField.Wavenumber) * r
        if isinstance(incidentField, MeshTransversalField): 
            return MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                incidentField.FieldX, 
                incidentField.FieldY, 
                incidentField.Wavefront + lensFunction, 
                incidentField.Domain)
        elif isinstance(incidentField, MeshElectricField): 
            return MeshElectricField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                incidentField.FieldX, 
                incidentField.FieldY,
                incidentField.FieldZ, 
                incidentField.Wavefront + lensFunction, 
                incidentField.Domain)
        elif isinstance(incidentField, RegularTransversalField): 
            return RegularTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                incidentField.FieldX * np.exp(1.0j * lensFunction), 
                incidentField.FieldY * np.exp(1.0j * lensFunction), 
                incidentField.Domain)
        elif isinstance(incidentField, RegularElectricField): 
            return RegularElectricField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                incidentField.FieldX * np.exp(1.0j * lensFunction), 
                incidentField.FieldY * np.exp(1.0j * lensFunction), 
                incidentField.FieldZ * np.exp(1.0j * lensFunction), 
                incidentField.Domain)
        else: 
            raise TypeError()
        
class CircularAperture:
    def __init__(
            self, 
            apertureRadius: float
    ): 
        if not isinstance(apertureRadius, float): 
            raise TypeError()
        if apertureRadius < 0.0: 
            raise ValueError()
        self._apertureRadius = apertureRadius
    
    @property
    def ApertureRadius(self): 
        return self._apertureRadius
    @ApertureRadius.setter
    def ApertureRadius(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        if value < 0.0:
            raise ValueError()
        self._apertureRadius = value
    
    @property
    def ApertureDiameter(self): 
        return 2.0 * self.ApertureRadius
    
    def PropagateFieldThrough(self, incidentField: FieldContainer): 
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()
        if incidentField.Domain != Domains.X:
            raise ValueError()
        
        rho = np.sqrt(
            np.power(incidentField.X, 2) + 
            np.power(incidentField.Y, 2), dtype=float)
        fieldX = np.where(rho <= self.ApertureRadius, incidentField.FieldX, np.zeros_like(incidentField.FieldX))
        fieldY = np.where(rho <= self.ApertureRadius, incidentField.FieldY, np.zeros_like(incidentField.FieldY))
        
        if isinstance(incidentField, MeshTransversalField): 
            return MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                fieldX, fieldY, 
                incidentField.Wavefront, 
                incidentField.Domain)
        elif isinstance(incidentField, MeshElectricField): 
            return MeshElectricField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                fieldX, 
                fieldY, 
                np.where(rho <= self.ApertureRadius, incidentField.FieldZ, np.zeros_like(incidentField.FieldZ)), 
                incidentField.Domain)
        if isinstance(incidentField, RegularTransversalField): 
            return RegularTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                fieldX, fieldY, 
                incidentField.Domain)
        if isinstance(incidentField, RegularElectricField): 
            return RegularElectricField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                fieldX, fieldY, 
                np.where(rho <= self.ApertureRadius, incidentField.FieldZ, np.zeros_like(incidentField.FieldZ)), 
                incidentField.Domain)
        else: 
            raise TypeError()

class Apodiser:
    def __init__(
            self, 
            apertureRadius: float, 
            edge: float
    ): 
        if not isinstance(apertureRadius, float): 
            raise TypeError()
        if apertureRadius < 0.0: 
            raise ValueError()
        if not isinstance(edge, float): 
            raise TypeError()
        if edge < 0.0: 
            raise ValueError()
        self._apertureRadius = apertureRadius
        self._edge = edge
    
    @property
    def ApertureRadius(self): 
        return self._apertureRadius
    @ApertureRadius.setter
    def ApertureRadius(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        if value < 0.0: 
            raise ValueError()
        self._apertureRadius = value
    
    @property
    def Edge(self): 
        return self._edge
    @Edge.setter
    def Edge(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        if value < 0.0: 
            raise ValueError()
        self._edge = value
    
    @property
    def ApertureDiameter(self): 
        return 2.0 * self.ApertureRadius
    
    @property
    def TotalApertureRadius(self): 
        return self.ApertureRadius + self.Edge
    
    @property
    def TotalApertureDiameter(self): 
        return 2.0 * self.TotalApertureRadius
    
    def PropagateFieldThrough(self, incidentField: FieldContainer): 
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()
        if incidentField.Domain != Domains.X: 
            raise ValueError()
        
        rho = np.sqrt(
            np.power(incidentField.X, 2) + 
            np.power(incidentField.Y, 2), dtype=float)
        apodisingMask = np.zeros_like(incidentField.X, dtype=float)
        apodisingMask = 0.5 * np.cos(np.pi * (rho - self.ApertureRadius) / self.Edge) + 0.5
        apodisingMask[rho <= self.ApertureRadius] = 1.0
        apodisingMask[rho > self.ApertureRadius + self.Edge] = 0.0

        if isinstance(incidentField, MeshTransversalField): 
            return MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                apodisingMask * incidentField.FieldX, 
                apodisingMask * incidentField.FieldY, 
                incidentField.Wavefront, 
                incidentField.Domain)
        if isinstance(incidentField, MeshElectricField): 
            return MeshElectricField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                apodisingMask * incidentField.FieldX, 
                apodisingMask * incidentField.FieldY,
                apodisingMask * incidentField.FieldZ,  
                incidentField.Wavefront, 
                incidentField.Domain)
        if isinstance(incidentField, RegularTransversalField): 
            return RegularTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                apodisingMask * incidentField.FieldX, 
                apodisingMask * incidentField.FieldY, 
                incidentField.Domain)
        if isinstance(incidentField, RegularElectricField): 
            return RegularElectricField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                apodisingMask * incidentField.FieldX, 
                apodisingMask * incidentField.FieldY,
                apodisingMask * incidentField.FieldZ, 
                incidentField.Domain)
        
class IdealPolariser: 
    def __init__(
            self, 
            angle: float
    ): 
        if not isinstance(angle, float): 
            raise TypeError()
        self._angle = angle
    
    @property
    def Angle(self): 
        return self._angle
    @Angle.setter
    def Angle(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._angle = value
    
    def PropagateFieldThrough(self, incidentField: FieldContainer): 
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()

        jonesMatrix = np.array([
            [np.power(np.cos(self.Angle),2), np.cos(self.Angle)*np.sin(self.Angle)], 
            [np.cos(self.Angle)*np.sin(self.Angle), np.power(np.sin(self.Angle),2)]
        ])

        if isinstance(incidentField, MeshTransversalField): 
            return MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                jonesMatrix[0,0] * incidentField.FieldX + jonesMatrix[0,1] * incidentField.FieldY, 
                jonesMatrix[1,0] * incidentField.FieldX + jonesMatrix[1,1] * incidentField.FieldY, 
                incidentField.Wavefront, 
                incidentField.Domain)
        elif isinstance(incidentField, RegularTransversalField): 
            return RegularTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                jonesMatrix[0,0] * incidentField.FieldX + jonesMatrix[0,1] * incidentField.FieldY, 
                jonesMatrix[1,0] * incidentField.FieldX + jonesMatrix[1,1] * incidentField.FieldY,  
                incidentField.Domain)
        else: 
            raise TypeError()

class IdealRetarder: 
    def __init__(
            self, 
            orientation: float, 
            phaseDelay: float
    ): 
        if not isinstance(orientation, float): 
            raise TypeError()
        if not isinstance(phaseDelay, float): 
            raise TypeError()
        self._orientation = orientation
        self._phaseDelay = phaseDelay

    @property
    def Orientation(self): 
        return self._orientation
    @Orientation.setter
    def Orientation(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._orientation = value
    
    @property
    def PhaseDelay(self): 
        return self._phaseDelay
    @PhaseDelay.setter
    def PhaseDelay(self, value): 
        if not isinstance(value, float): 
            raise TypeError()
        self._phaseDelay = value

    @property
    def JonesMatrix(self): 
        jonesMatrix = np.array([
            [np.exp(-0.5j*self.PhaseDelay), 0.0], 
            [0.0, np.exp(0.5j*self.PhaseDelay)]
        ])
        rotationMatrix = np.array([
            [np.cos(self.Orientation), -np.sin(self.Orientation)], 
            [np.sin(self.Orientation), np.cos(self.Orientation)]
        ])
        jonesMatrix = rotationMatrix @ jonesMatrix @ rotationMatrix.T
        return jonesMatrix

    
    def PropagateFieldThrough(self, incidentField: FieldContainer): 
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()

        jonesMatrix = np.array([
            [np.exp(-0.5j*self.PhaseDelay), 0.0], 
            [0.0, np.exp(0.5j*self.PhaseDelay)]
        ])
        rotationMatrix = np.array([
            [np.cos(self.Orientation), -np.sin(self.Orientation)], 
            [np.sin(self.Orientation), np.cos(self.Orientation)]
        ])
        jonesMatrix = rotationMatrix @ jonesMatrix @ rotationMatrix.T
        fieldX = jonesMatrix[0,0] * incidentField.FieldX + jonesMatrix[0,1] * incidentField.FieldY
        fieldY = jonesMatrix[1,0] * incidentField.FieldX + jonesMatrix[1,1] * incidentField.FieldY

        if isinstance(incidentField, MeshTransversalField): 
            return MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                fieldX, fieldY, 
                incidentField.Wavefront, 
                incidentField.Domain)
        if isinstance(incidentField, RegularTransversalField): 
            return RegularTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                fieldX, fieldY,
                incidentField.Domain)
        else: 
            raise TypeError()
        
class IdealRotator: 
    def __init__(
            self, 
            orientation: float,
            rotationAngle: float
    ): 
        if not isinstance(rotationAngle, float): 
            raise TypeError()
        if not isinstance(orientation, float): 
            raise TypeError()
        self._rotationAngle = rotationAngle
        self._orientation = orientation
    
    @property
    def RotationAngle(self): 
        return self._rotationAngle
    @RotationAngle.setter
    def RotationAngle(self, value: float): 
        if not isinstance(value, float): 
            raise TypeError()
        self._rotationAngle = value
    
    @property
    def Orientation(self): 
        return self._orientation
    @Orientation.setter
    def Orientation(self, value: float): 
        if not isinstance(value, float): 
            raise TypeError()
        self._orientation = value
    
    @property
    def JonesMatrix(self): 
        jonesMatrix = np.array([
            [np.cos(self.RotationAngle), -np.sin(self.RotationAngle)], 
            [np.sin(self.RotationAngle), np.cos(self.RotationAngle)]])
        rotationMatrix = np.array([
            [np.cos(self.Orientation), -np.sin(self.Orientation)], 
            [np.sin(self.Orientation), np.cos(self.Orientation)]
        ])
        jonesMatrix = rotationMatrix @ jonesMatrix @ rotationMatrix.T
        return jonesMatrix
    
    def PropagateFieldThrough(self, incidentField: FieldContainer): 
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()
        
        if isinstance(incidentField, MeshTransversalField):
            return MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                self.JonesMatrix[0,0] * incidentField.FieldX + self.JonesMatrix[0,1] * incidentField.FieldY, 
                self.JonesMatrix[1,0] * incidentField.FieldX + self.JonesMatrix[1,1] * incidentField.FieldY, 
                incidentField.Wavefront, 
                incidentField.Domain)
        elif isinstance(incidentField, RegularTransversalField): 
            return RegularTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                self.JonesMatrix[0,0] * incidentField.FieldX + self.JonesMatrix[0,1] * incidentField.FieldY, 
                self.JonesMatrix[1,0] * incidentField.FieldX + self.JonesMatrix[1,1] * incidentField.FieldY, 
                incidentField.Domain)
        else: 
            raise TypeError(f'Error in {IdealRotator.__name__}.{IdealRotator.PropagateFieldThrough.__name__} → Propagation through Jones matrix transmission function only implemented for {MeshTransversalField.__name__} and {RegularTransversalField.__name__}.')
    
class IdealNonParaxialLens:
    def __init__(
            self, 
            focalLength: float, 
            dispersionBehind: Material): 
        if not isinstance(focalLength, float): 
            raise TypeError()
        if not isinstance(dispersionBehind, Material):
            raise TypeError()
        self._focalLength = focalLength
        self._dispersionBehind = dispersionBehind

    @property
    def FocalLength(self): 
        return self._focalLength
    @FocalLength.setter
    def FocalLength(self, value): 
        self._focalLength = value

    @property
    def DispersionBehind(self): 
        return self._dispersionBehind
    @DispersionBehind.setter
    def DispersionBehind(self, value): 
        if not isinstance(value, Material):
            raise TypeError()
        self._dispersionBehind = value

    def PropagateFieldThrough(self, incidentField: FieldContainer): 
        if not isinstance(incidentField, FieldContainer):
            raise TypeError()
        
        field = copy.deepcopy(incidentField)
        field.Dispersion = self.DispersionBehind
        field = IdealParaxialLens(self.FocalLength).PropagateFieldThrough(incidentField)

        fieldk = FourierTransforms.PFT(field)
        k = np.zeros((np.shape(fieldk.X)[0], 3), dtype=float)
        k[:,0] = np.real(fieldk.X)
        k[:,1] = np.real(fieldk.Y)
        k[:,2] = np.real(fieldk.Kz)

        geometryFactor = np.where(
            np.real(fieldk.Kz) != 0, 
            np.sqrt(incidentField.Wavenumber / np.real(fieldk.Kz)), 
            1.0)
        fieldk.FieldX = geometryFactor * fieldk.FieldX
        fieldk.FieldY = geometryFactor * fieldk.FieldY
        field = FourierTransforms.IPFT(fieldk)
        return field
        
        


        

        
