# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .FieldContainers import FieldContainer, RegularTransversalField, RegularElectricField

class Detectors: 
    @staticmethod
    def FieldOnDetector(field: FieldContainer, detectorWindow: float, detectorPixels: int): 
        if not isinstance(field, FieldContainer): 
            raise TypeError(f'Error in {Detectors.__name__}.{Detectors.FieldOnDetector.__name__} → The field must be an instance of a class derived from {FieldContainer.__name__}.')
        if not isinstance(detectorWindow, float): 
            raise TypeError(f'Error in {Detectors.__name__}.{Detectors.FieldOnDetector.__name__} → The detector window must be a float.')
        if detectorWindow <= 0.0: 
            raise ValueError(f'Error in {Detectors.__name__}.{Detectors.FieldOnDetector.__name__} → The detector window must be positive.')
        if not isinstance(detectorPixels, int): 
            raise TypeError(f'Error in {Detectors.__name__}.{Detectors.FieldOnDetector.__name__} → The number of detector pixels must be an int.')
        if detectorPixels <= 0: 
            raise ValueError(f'Error in {Detectors.__name__}.{Detectors.FieldOnDetector.__name__} → The number of detector pixels must be positive.')
        
        if isinstance(field, RegularTransversalField): 
            x = field.X[:, 0]
            y = field.Y[0, :]

            interpolatorFieldX = RegularGridInterpolator((x, y), field.FieldX, method='linear', bounds_error=False, fill_value=0.0)
            interpolatorFieldY = RegularGridInterpolator((x, y), field.FieldY, method='linear', bounds_error=False, fill_value=0.0)

            xDetector = np.linspace(-0.5*detectorWindow, 0.5*detectorWindow, detectorPixels)
            yDetector = np.linspace(-0.5*detectorWindow, 0.5*detectorWindow, detectorPixels)
            xDetector, yDetector = np.meshgrid(xDetector, yDetector, indexing='ij')

            fieldX = interpolatorFieldX(np.column_stack([xDetector.ravel(), yDetector.ravel()]))
            fieldY = interpolatorFieldY(np.column_stack([xDetector.ravel(), yDetector.ravel()]))

            fieldX = fieldX.reshape(xDetector.shape)
            fieldY = fieldY.reshape(xDetector.shape)

            return RegularTransversalField(
                field.Wavelength, 
                field.Dispersion, 
                xDetector, yDetector, 
                fieldX, fieldY, 
                field.Domain)
        
        elif isinstance(field, RegularElectricField): 
            x = field.X[:, 0]
            y = field.Y[0, :]

            interpolatorFieldX = RegularGridInterpolator((x, y), field.FieldX, method='linear', bounds_error=False, fill_value=0.0)
            interpolatorFieldY = RegularGridInterpolator((x, y), field.FieldY, method='linear', bounds_error=False, fill_value=0.0)
            interpolatorFieldZ = RegularGridInterpolator((x, y), field.FieldZ, method='linear', bounds_error=False, fill_value=0.0)

            xDetector = np.linspace(-0.5*detectorWindow, 0.5*detectorWindow, detectorPixels)
            yDetector = np.linspace(-0.5*detectorWindow, 0.5*detectorWindow, detectorPixels)
            xDetector, yDetector = np.meshgrid(xDetector, yDetector, indexing='ij')

            fieldX = interpolatorFieldX(np.column_stack([xDetector.ravel(), yDetector.ravel()]))
            fieldY = interpolatorFieldY(np.column_stack([xDetector.ravel(), yDetector.ravel()]))
            fieldZ = interpolatorFieldZ(np.column_stack([xDetector.ravel(), yDetector.ravel()]))

            fieldX = fieldX.reshape(xDetector.shape)
            fieldY = fieldY.reshape(xDetector.shape)
            fieldZ = fieldZ.reshape(xDetector.shape)

            return RegularElectricField(
                field.Wavelength, 
                field.Dispersion, 
                xDetector, yDetector, 
                fieldX, fieldY, fieldZ, 
                field.Domain)

        
        else: 
            raise TypeError('Not implemented yet!')
