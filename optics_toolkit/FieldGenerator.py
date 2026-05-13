# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.special import genlaguerre

from .FieldContainers import Sampling, Domains, MeshTransversalField
from .Materials import Material

class FieldGenerator: 
    @staticmethod
    def LaguerreGaussAtWaist_Mesh(
        wavelength: float, 
        dispersion: Material, 
        waistRadius: float, 
        radialOrder: int, 
        azimuthalOrder: int, 
        jonesVector: np.ndarray, 
        windowSize: float, 
        meshLevels: int
    ): 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not isinstance(dispersion, Material): 
            raise TypeError()
        if not dispersion.IsWavelengthWithinRange(wavelength): 
            raise ValueError()
        if not isinstance(waistRadius, float): 
            raise TypeError()
        if waistRadius <= 0.0: 
            raise ValueError()
        if not isinstance(radialOrder, int): 
            raise TypeError()
        if radialOrder < 0: 
            raise ValueError()
        if not isinstance(azimuthalOrder, int): 
            raise TypeError()
        if not isinstance(jonesVector, np.ndarray): 
            raise TypeError()
        if not (np.issubdtype(jonesVector.dtype, float) or np.issubdtype(jonesVector.dtype, complex)): 
            raise TypeError()
        if not isinstance(windowSize, float): 
            raise TypeError()
        if windowSize <= 0.0: 
            raise ValueError()
        if not isinstance(meshLevels, int): 
            raise TypeError()
        if meshLevels <= 0: 
            raise ValueError()
        
        xmesh, ymesh = Sampling.ConstructMesh(0.5*windowSize, meshLevels)

        rho = np.sqrt(np.power(xmesh, 2) + np.power(ymesh, 2))
        phi = np.arctan2(ymesh, xmesh)

        field = (
            (1.0 / waistRadius) * 
            np.power(rho * np.sqrt(2.0) / waistRadius, np.abs(azimuthalOrder)) * 
            np.exp(-np.power(rho, 2) / np.power(waistRadius, 2)) * 
            genlaguerre(radialOrder, np.abs(azimuthalOrder))(2.0 * np.power(rho, 2) / np.power(waistRadius, 2)) * 
            np.exp(-1.0j * azimuthalOrder * phi)
        )
        field = field / np.nanmax(field)

        return MeshTransversalField(
            wavelength, 
            dispersion, 
            xmesh, ymesh, 
            jonesVector[0] * field, 
            jonesVector[1] * field, 
            np.zeros_like(xmesh), 
            Domains.X)


