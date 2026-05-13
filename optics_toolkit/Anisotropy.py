# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
from enum import Enum
import pyvista as pv

from .Materials import Material
from .FieldContainers import Domains, FieldContainer, MeshTransversalField

class AnisotropySupportFunctions: 
    @staticmethod
    def CalculateOmegaMatrix(kxNorm: float, kyNorm: float, permittivityTensor: np.ndarray, permeabilityTensor: np.ndarray):
        # if not isinstance(kxNorm, float):
        #     raise TypeError('The normalised kx component should be a float.')
        # if not isinstance(kyNorm, float):
        #     raise TypeError('The normalised ky component should be a float.')
        if not isinstance(permittivityTensor, np.ndarray):
            raise TypeError('The permittivity tensor should be delivered in numpy array form.')
        if not isinstance(permeabilityTensor, np.ndarray):
            raise TypeError('The permeability tensor should be delivered in numpy array form.')
        if np.shape(permittivityTensor) != (3,3):
            raise ValueError('The dimensions of the permittivity tensor should be 3x3.')
        if np.shape(permeabilityTensor) != (3,3):
            raise ValueError('The dimensions of the permeability tensor should be 3x3.')
        omega = np.zeros((4, 4), dtype=complex)
        omega[0,0] = -kxNorm * permittivityTensor[2,0]/permittivityTensor[2,2] - kyNorm * permeabilityTensor[1,2] / permeabilityTensor[2,2]
        omega[0,1] = -kxNorm * (permittivityTensor[2,1]/permittivityTensor[2,2] - permeabilityTensor[1,2]/permeabilityTensor[2,2])
        omega[0,2] = permeabilityTensor[1,0] - permeabilityTensor[1,2]*permeabilityTensor[2,0]/permeabilityTensor[2,2] + kxNorm*kyNorm/permittivityTensor[2,2]
        omega[0,3] = permeabilityTensor[1,1] - permeabilityTensor[1,2]*permeabilityTensor[2,1]/permeabilityTensor[2,2] - kxNorm*kxNorm/permittivityTensor[2,2]

        omega[1,0] = -kyNorm * (permittivityTensor[2,0]/permittivityTensor[2,2] - permeabilityTensor[0,2]/permeabilityTensor[2,2])
        omega[1,1] = -kyNorm * permittivityTensor[2,1]/permittivityTensor[2,2] - kxNorm * permeabilityTensor[0,2]/permeabilityTensor[2,2]
        omega[1,2] = -permeabilityTensor[0,0] + permeabilityTensor[0,2]*permeabilityTensor[2,0]/permeabilityTensor[2,2] + kyNorm*kyNorm/permittivityTensor[2,2]
        omega[1,3] = -permeabilityTensor[0,1] + permeabilityTensor[0,2]*permeabilityTensor[2,1]/permeabilityTensor[2,2] - kxNorm*kyNorm/permittivityTensor[2,2]

        omega[2,0] = -permittivityTensor[1,0] + permittivityTensor[1,2]*permittivityTensor[2,0]/permittivityTensor[2,2] - kxNorm*kyNorm/permeabilityTensor[2,2]
        omega[2,1] = -permittivityTensor[1,1] + permittivityTensor[1,2]*permittivityTensor[2,1]/permittivityTensor[2,2] + kxNorm*kxNorm/permeabilityTensor[2,2]
        omega[2,2] = -kyNorm*permittivityTensor[1,2]/permittivityTensor[2,2] - kxNorm*permeabilityTensor[2,0]/permeabilityTensor[2,2]
        omega[2,3] = kxNorm*(permittivityTensor[1,2]/permittivityTensor[2,2] - permeabilityTensor[2,1]/permeabilityTensor[2,2])

        omega[3,0] = permittivityTensor[0,0] - permittivityTensor[2,0]*permittivityTensor[0,2]/permittivityTensor[2,2] - kyNorm*kyNorm/permeabilityTensor[2,2]
        omega[3,1] = permittivityTensor[0,1] - permittivityTensor[2,1]*permittivityTensor[0,2]/permittivityTensor[2,2] + kxNorm*kyNorm/permeabilityTensor[2,2]
        omega[3,2] = kyNorm * (permittivityTensor[0,2]/permittivityTensor[2,2] - permeabilityTensor[2,0]/permeabilityTensor[2,2])
        omega[3,3] = -kxNorm*permittivityTensor[0,2]/permittivityTensor[2,2] - kyNorm*permeabilityTensor[2,1]/permeabilityTensor[2,2]

        return omega

    @staticmethod
    def SolveEigenProblem(omega: np.ndarray):
        
        values, vectors = np.linalg.eig(omega)

        indices_values_pos = np.where(np.real(values) > .0)[0]

        values_pos = values[indices_values_pos]
        vectors_pos = vectors[:, indices_values_pos] 
        sorted_pos = np.argsort(np.real(values_pos))
        values_pos = values_pos[sorted_pos]
        vectors_pos = vectors_pos[:, sorted_pos]

        indices_values_neg = np.where(np.real(values) < .0)[0]
        values_neg = values[indices_values_neg]
        vectors_neg = vectors[:, indices_values_neg]
        sorted_neg = np.argsort(np.real(values_neg))
        values_neg = values_neg[sorted_neg]
        vectors_neg = vectors_neg[:, sorted_neg]

        values = np.hstack((values_pos, values_neg))
        vectors = np.hstack((vectors_pos, vectors_neg))

        return values, vectors
    
    @staticmethod 
    def SolveEigenProblemIsotropic(kxNorm: float, kyNorm: float, permittivity, permeability):
        if not (isinstance(kxNorm, float) or isinstance(kxNorm, complex)):
            raise TypeError('The normalised kx component should be a float.')
        if not (isinstance(kyNorm, float) or isinstance(kyNorm, complex)):
            raise TypeError('The normalised ky component should be a float.')
        eigenvalue = np.sqrt(permittivity * permeability - kxNorm**2 - kyNorm**2)
        values = np.array([eigenvalue, eigenvalue, -eigenvalue, -eigenvalue])
        wb = (kxNorm * kyNorm / permeability) / (-permittivity + kxNorm*kxNorm/permeability)
        wc = eigenvalue / (permeability - (kxNorm*kxNorm/permittivity))
        wd = eigenvalue / (-permittivity + kxNorm*kxNorm/permeability)
        we = (-kxNorm * kyNorm / permittivity) / (permeability - kxNorm*kxNorm/permittivity)
        vectors = np.array([
            [1.0, .0, 1.0, .0], 
            [wb, wd, wb, -wd], 
            [.0, 1.0, .0, 1.0], 
            [wc, we, -wc, we]
        ])
        return values, vectors

    @staticmethod
    def OpticAxisAngleOfBiaxialCrystal(nxx: float, nyy: float, nzz: float):
        if not isinstance(nxx, float):
            raise TypeError('nxx should be a float.')
        if not isinstance(nyy, float):
            raise TypeError('nyy should be a float.')
        if not isinstance(nzz, float):
            raise TypeError('nzz should be a float.')
        
        if nxx == nyy or nyy == nzz: 
            raise ValueError('The crystal has to be biaxial.')
        
        angle = np.sqrt((np.power(nxx,-2) - np.power(nyy,-2)) / (np.power(nyy,-2) - np.power(nzz,-2)))
        angle = np.arctan(angle)
        return angle
    
    @staticmethod
    def AngleOfConicalRefractionCone(nxx: float, nyy: float, nzz: float):
        if not isinstance(nxx, float):
            raise TypeError('nxx should be a float.')
        if not isinstance(nyy, float):
            raise TypeError('nyy should be a float.')
        if not isinstance(nzz, float):
            raise TypeError('nzz should be a float.')
        
        if nxx == nyy or nyy == nzz:
            raise ValueError('The crystal has to be biaxial.')
        
        angle = 0.5 * np.arctan(
            np.power(nyy,2) * 
            np.sqrt((np.power(nxx,-2) - np.power(nyy,-2)) * (np.power(nyy,-2) - np.power(nzz,-2))))
        return angle
    
    @staticmethod 
    def LateralShiftConicalRefraction(nxx: float, nyy: float, nzz: float, thickness: float):
        if not isinstance(nxx, float):
            raise TypeError('nxx should be a float.')
        if not isinstance(nyy, float):
            raise TypeError('nyy should be a float.')
        if not isinstance(nzz, float):
            raise TypeError('nzz should be a float.')
        if not isinstance(thickness, float):
            raise TypeError('The thickness should be a float.')
        if thickness <= 0.0:
            raise ValueError('The thickness should be positive.')
        
        if nxx == nyy or nyy == nzz:
            raise ValueError('The crystal needs to be biaxial.')
        
        shift = 0.5 * thickness * np.tan(2.0 * AnisotropySupportFunctions.AngleOfConicalRefractionCone(nxx, nyy, nzz))
        return shift
    
    @staticmethod
    def sMatrix(vectors_1, vectors_2):
        W_left = np.hstack((vectors_2[:, :2], -vectors_1[:, 2:]))
        W_right = np.hstack((vectors_1[:, :2], -vectors_2[:, 2:]))
        s_matrix = np.linalg.inv(W_left) @ W_right
        return s_matrix
    
    @staticmethod
    def sMatrix_PlusPlus(vectors_1, vectors_2):
        s = AnisotropySupportFunctions.sMatrix(vectors_1, vectors_2)
        return s[:2, :2]
    
    @staticmethod
    def sMatrix_PlusMinus(vectors_1, vectors_2):
        s = AnisotropySupportFunctions.sMatrix(vectors_1, vectors_2)
        return s[:2, 2:]
    
    @staticmethod
    def sMatrix_MinusPlus(vectors_1, vectors_2):
        s = AnisotropySupportFunctions.sMatrix(vectors_1, vectors_2)
        return s[2:, :2]
    
    @staticmethod
    def sMatrix_MinusMinus(vectors_1, vectors_2):
        s = AnisotropySupportFunctions.sMatrix(vectors_1, vectors_2)
        return s[2:, 2:]

class BiaxialCrystal:
    def __init__(
            self, 
            dispersionX: Material, 
            dispersionY: Material, 
            dispersionZ: Material
    ): 
        if not isinstance(dispersionX, Material): 
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.__init__.__name__} → The dispersion characteristics of the refractive indices of the diagonal tensor of the crystal must be given as an instance of a class derived from {Material.__name__}.')
        if not isinstance(dispersionY, Material):
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.__init__.__name__} → The dispersion characteristics of the refractive indices of the diagonal tensor of the crystal must be given as an instance of a class derived from {Material.__name__}.')
        if not isinstance(dispersionZ, Material): 
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.__init__.__name__} → The dispersion characteristics of the refractive indices of the diagonal tensor of the crystal must be given as an instance of a class derived from {Material.__name__}.')
        self._dispersionX = dispersionX
        self._dispersionY = dispersionY
        self._dispersionZ = dispersionZ

    @property
    def DispersionX(self) -> Material: 
        return self._dispersionX
    @DispersionX.setter
    def DispersionX(self, value): 
        if not isinstance(value, Material): 
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.__init__.__name__} → The dispersion characteristics of the refractive indices of the diagonal tensor of the crystal must be given as an instance of a class derived from {Material.__name__}.')
        self._dispersionX = value
    
    @property
    def DispersionY(self) -> Material: 
        return self._dispersionY
    @DispersionY.setter
    def DispersionY(self, value): 
        if not isinstance(value, Material): 
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.__init__.__name__} → The dispersion characteristics of the refractive indices of the diagonal tensor of the crystal must be given as an instance of a class derived from {Material.__name__}.')
        self._dispersionY = value
    
    @property
    def DispersionZ(self) -> Material: 
        return self._dispersionZ
    @DispersionZ.setter
    def DispersionZ(self, value): 
        if not isinstance(value, Material): 
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.__init__.__name__} → The dispersion characteristics of the refractive indices of the diagonal tensor of the crystal must be given as an instance of a class derived from {Material.__name__}.')
        self._dispersionZ = value
    
    @property
    def MinimumWavelength(self) -> float: 
        return np.nanmax((
            self.DispersionX.MinimumWavelengthOfDispersion, 
            self.DispersionY.MinimumWavelengthOfDispersion, 
            self.DispersionZ.MinimumWavelengthOfDispersion))
    
    @property
    def MaximumWavelength(self) -> float: 
        return np.nanmin((
            self.DispersionX.MaximumWavelengthOfDispersion, 
            self.DispersionY.MaximumWavelengthOfDispersion, 
            self.DispersionZ.MaximumWavelengthOfDispersion))
    
    def IsWavelengthInDispersionRange(self, wavelength: float) -> bool: 
        if not isinstance(wavelength, float): 
            raise TypeError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.IsWavelengthInDispersionRange.__name__} → The wavelength must be a float.')
        if wavelength <= 0.0: 
            raise ValueError(f'Error in {BiaxialCrystal.__name__}.{BiaxialCrystal.IsWavelengthInDispersionRange.__name__} → The wavelength must be positive.')
        if wavelength < self.MinimumWavelength or wavelength > self.MaximumWavelength: 
            return False
        else: 
            return True
    
    def RelativePermittivityTensor(self, wavelength: float) -> np.ndarray: 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        
        exx = self.DispersionX.RelativePermittivity.GetValue(wavelength)
        eyy = self.DispersionY.RelativePermittivity.GetValue(wavelength)
        ezz = self.DispersionZ.RelativePermittivity.GetValue(wavelength)

        return np.array([
            [exx, 0.0, 0.0], 
            [0.0, eyy, 0.0], 
            [0.0, 0.0, ezz]])
    
    def RelativePermeabilityTensor(self, wavelength: float) -> np.ndarray: 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        
        mxx = self.DispersionX.RelativePermeability.GetValue(wavelength)
        myy = self.DispersionY.RelativePermeability.GetValue(wavelength)
        mzz = self.DispersionZ.RelativePermeability.GetValue(wavelength)

        return np.array([
            [mxx, 0.0, 0.0], 
            [0.0, myy, 0.0], 
            [0.0, 0.0, mzz]])
    
    def CalculateAngleOfOpticAxes(self, wavelength: float) -> float: 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        nxx = self.DispersionX.RefractiveIndex.GetRealValue(wavelength)
        nyy = self.DispersionY.RefractiveIndex.GetRealValue(wavelength)
        nzz = self.DispersionZ.RefractiveIndex.GetRealValue(wavelength)

        refractiveIndices = np.array([nxx, nyy, nzz])
        sortedIndices = np.argsort(np.real(refractiveIndices))
        refractiveIndices = refractiveIndices[sortedIndices]
        angle = AnisotropySupportFunctions.OpticAxisAngleOfBiaxialCrystal(
            refractiveIndices[0], 
            refractiveIndices[1], 
            refractiveIndices[2])
        return angle
    
    def CalculateAngleOfConicalRefractionCone(self, wavelength: float) -> float: 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        nxx = self.DispersionX.RefractiveIndex.GetRealValue(wavelength)
        nyy = self.DispersionY.RefractiveIndex.GetRealValue(wavelength)
        nzz = self.DispersionZ.RefractiveIndex.GetRealValue(wavelength)

        refractiveIndices = np.array([nxx, nyy, nzz])
        sortedIndices = np.argsort(np.real(refractiveIndices))
        refractiveIndices = refractiveIndices[sortedIndices]
        angle = AnisotropySupportFunctions.AngleOfConicalRefractionCone(
            refractiveIndices[0], 
            refractiveIndices[1], 
            refractiveIndices[2])
        return angle
    
class CrystalPlate: 
    def PropagateFieldThrough(self, incidentField: FieldContainer):
        if not isinstance(incidentField, FieldContainer): 
            raise TypeError()
        if incidentField.Domain != Domains.K: 
            raise ValueError()
        
        mode1_Ex = np.zeros_like(incidentField.FieldX, dtype=complex)
        mode1_Ey = np.zeros_like(incidentField.FieldY, dtype=complex)
        mode2_Ex = np.zeros_like(incidentField.FieldX, dtype=complex)
        mode2_Ey = np.zeros_like(incidentField.FieldY, dtype=complex)

        if isinstance(incidentField, MeshTransversalField): 
            mode1_wavefront = np.zeros_like(incidentField.Wavefront, dtype=float)
            mode2_wavefront = np.zeros_like(incidentField.Wavefront, dtype=float)

            for i in range (0, len(incidentField.X)): 
                kx = incidentField.X[i]
                ky = incidentField.Y[i]

                nx = kx / incidentField.K0
                ny = ky / incidentField.K0

                values_isotropic, vectors_isotropic = AnisotropySupportFunctions.SolveEigenProblemIsotropic(
                    nx, ny, 
                    incidentField.Dispersion.RelativePermittivity.GetValue(incidentField.Wavelength), 
                    incidentField.Dispersion.RelativePermeability.GetValue(incidentField.Wavelength))
                values_crystal, vectors_crystal = AnisotropySupportFunctions.SolveEigenProblem(
                    AnisotropySupportFunctions.CalculateOmegaMatrix(
                        nx, ny, 
                        self.GetRelativePermittivityTensor(incidentField.Wavelength), 
                        self.GetRelativePermeabilityTensor(incidentField.Wavelength)))
                
                smatrix1 = AnisotropySupportFunctions.sMatrix_PlusPlus(vectors_isotropic, vectors_crystal)
                smatrix2 = AnisotropySupportFunctions.sMatrix_PlusPlus(vectors_crystal, vectors_isotropic)

                field_in = np.array([
                    incidentField.FieldX[i], incidentField.FieldY[i]])
                c = np.linalg.inv(vectors_isotropic[:2,:2]) @ field_in
                c = smatrix1 @ c

                c1 = np.array([c[0], 0.0])
                c2 = np.array([0.0, c[1]])

                c1 = smatrix2 @ c1
                c2 = smatrix2 @ c2

                mode1 = vectors_isotropic[:2,:2] @ c1
                mode2 = vectors_isotropic[:2,:2] @ c2

                mode1_Ex[i] = mode1[0]
                mode1_Ey[i] = mode1[1]
                mode2_Ex[i] = mode2[0]
                mode2_Ey[i] = mode2[1]

                mode1_wavefront[i] = incidentField.Wavefront[i] + incidentField.K0 * values_crystal[0] * self.Thickness
                mode2_wavefront[i] = incidentField.Wavefront[i] + incidentField.K0 * values_crystal[1] * self.Thickness

            spectrumMode1 = MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                mode1_Ex, mode1_Ey, 
                mode1_wavefront, 
                incidentField.Domain)
            spectrumMode2 = MeshTransversalField(
                incidentField.Wavelength, 
                incidentField.Dispersion, 
                incidentField.X, incidentField.Y, 
                mode2_Ex, mode2_Ey, 
                mode2_wavefront, 
                incidentField.Domain)
            return spectrumMode1, spectrumMode2

class BiaxialCrystalPlate(CrystalPlate): 
    def __init__(
            self, 
            crystal: BiaxialCrystal, 
            thickness: float, 
            rotationMatrix: np.ndarray
    ): 
        if not isinstance(crystal, BiaxialCrystal): 
            raise TypeError()
        if not isinstance(thickness, float): 
            raise TypeError()
        if thickness < 0.0: 
            raise ValueError()
        if not isinstance(rotationMatrix, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(rotationMatrix.dtype, float): 
            raise TypeError()
        if np.shape(rotationMatrix) != (3,3): 
            raise ValueError()
        assert(np.allclose(np.linalg.det(rotationMatrix), 1.0))
        self._crystal = crystal
        self._thickness = thickness
        self._rotationMatrix = rotationMatrix
    
    @property
    def Crystal(self) -> BiaxialCrystal:
        return self._crystal
    @Crystal.setter
    def Crystal(self, value: BiaxialCrystal): 
        if not isinstance(value, BiaxialCrystal): 
            raise TypeError()
        self._crystal = value
    
    @property
    def Thickness(self) -> float: 
        return self._thickness
    @Thickness.setter
    def Thickness(self, value: float): 
        if not isinstance(value, float): 
            raise TypeError()
        if value < 0.0: 
            raise ValueError()
        self._thickness = value
    
    @property
    def RotationMatrix(self) -> np.ndarray: 
        return self._rotationMatrix
    @RotationMatrix.setter
    def RotationMatrix(self, value: np.ndarray): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        if np.shape(value) != (3,3): 
            raise ValueError()
        assert(np.allclose(np.linalg.det(value), 1.0)) 
        self._rotationMatrix = value
    
    def GetRelativePermittivityTensor(self, wavelength: float) -> np.ndarray:
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.Crystal.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        
        return self.RotationMatrix @ self.Crystal.RelativePermittivityTensor(wavelength) @ self.RotationMatrix.T
    
    def GetRelativePermeabilityTensor(self, wavelength: float) -> np.ndarray:
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.Crystal.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        
        return self.RotationMatrix @ self.Crystal.RelativePermeabilityTensor(wavelength) @ self.RotationMatrix.T

    def GetOpticAxisVectors(self, wavelength: float) -> np.ndarray:
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.Crystal.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        
        opticAxisAngle = self.Crystal.CalculateAngleOfOpticAxes(wavelength)
        opticAxis1 = np.array([
            np.sin(opticAxisAngle), 0.0, np.cos(opticAxisAngle)])
        opticAxis2 = np.array([
            -np.sin(opticAxisAngle), 0.0, np.cos(opticAxisAngle)])
        opticAxis1 = self.RotationMatrix @ opticAxis1
        opticAxis2 = self.RotationMatrix @ opticAxis2
        opticAxes = np.column_stack([opticAxis1, opticAxis2])
        return opticAxes

    def CalculateLateralShiftOfConicalRefraction(self, wavelength: float): 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.Crystal.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        nxx = self.Crystal.DispersionX.RefractiveIndex.GetRealValue(wavelength)
        nyy = self.Crystal.DispersionY.RefractiveIndex.GetRealValue(wavelength)
        nzz = self.Crystal.DispersionZ.RefractiveIndex.GetRealValue(wavelength)
        return AnisotropySupportFunctions.LateralShiftConicalRefraction(
            nxx, nyy, nzz, self.Thickness)
        
    
    def DisplayCrystalGeometry(self, wavelength: float): 
        if not isinstance(wavelength, float): 
            raise TypeError()
        if wavelength <= 0.0: 
            raise ValueError()
        if not self.Crystal.IsWavelengthInDispersionRange(wavelength): 
            raise ValueError()
        
        # Create plotter
        plotter = pv.Plotter()

        # Create entrance plane of crystal plate
        plane = pv.Plane(
            center = (0, 0, 0), 
            direction = (0, 0, 1), 
            i_size = 5, 
            j_size = 5)
        plotter.add_mesh(plane, color='lightblue', opacity=0.5, label='Entrance Plane')

        # Create Cartesian basis (global CS): 
        xaxis = pv.Arrow(
            start = (0, 0, 0), 
            direction = (1, 0, 0), 
            scale = 2)
        plotter.add_mesh(xaxis, color='red', label='X Axis')
        yaxis = pv.Arrow(
            start = (0, 0, 0), 
            direction = (0, 1, 0), 
            scale = 2)
        plotter.add_mesh(yaxis, color='green', label='Y Axis')
        zaxis = pv.Arrow(
            start = (0, 0, 0), 
            direction = (0, 0, 1), 
            scale = 2)
        plotter.add_mesh(zaxis, color='blue', label='Z Axis')

        # Get crystallography basis
        principalAxis1 = pv.Arrow(
            start = (0, 0, 0), 
            direction = (self.RotationMatrix[:,0]), 
            scale = 2)
        plotter.add_mesh(principalAxis1, color='cyan', label='Principal Axis 1')
        principalAxis2 = pv.Arrow(
            start = (0, 0, 0), 
            direction = (self.RotationMatrix[:,1]), 
            scale = 2)
        plotter.add_mesh(principalAxis2, color='magenta', label='Principal Axis 2')
        principalAxis3 = pv.Arrow(
            start = (0, 0, 0), 
            direction = (self.RotationMatrix[:,2]), 
            scale = 2)
        plotter.add_mesh(principalAxis3, color='gold', label='Principal Axis 3')
        
        # Add optic axes
        opticAxisAngle = self.Crystal.CalculateAngleOfOpticAxes(wavelength)
        opticAxis1 = np.array([np.sin(opticAxisAngle), 0.0, np.cos(opticAxisAngle)])
        opticAxis2 = np.array([-np.sin(opticAxisAngle), 0.0, np.cos(opticAxisAngle)])
        opticAxis1 = self.RotationMatrix @ opticAxis1
        opticAxis2 = self.RotationMatrix @ opticAxis2
        opticAxis1 = pv.Arrow(
            start = (0, 0, 0), 
            direction = (opticAxis1), 
            scale = 2)
        opticAxis2 = pv.Arrow(
            start = (0, 0, 0), 
            direction = (opticAxis2), 
            scale = 2)
        plotter.add_mesh(opticAxis1, color='black', label='Optic Axis 1')
        plotter.add_mesh(opticAxis2, color='grey', label='Optic Axis 2')
        plotter.add_title(f'Biaxial Crystal Geometry')
        plotter.add_legend()
        plotter.show()

    def CalculateRealPolarAngleAndShift(self, wavelength: float): 
        angleOfOpticAxis = self.Crystal.CalculateAngleOfOpticAxes(wavelength)
        opticAxis1 = np.array([np.sin(angleOfOpticAxis), 0.0, np.cos(angleOfOpticAxis)])
        opticAxis2 = np.array([-np.sin(angleOfOpticAxis), 0.0, np.cos(angleOfOpticAxis)])

        opticAxis1 = self.RotationMatrix @ opticAxis1
        opticAxis2 = self.RotationMatrix @ opticAxis2

        lateralShift = self.CalculateLateralShiftOfConicalRefraction(wavelength)

        if np.abs(opticAxis1[2]) > np.abs(opticAxis2[2]):
            realPolarAngle = np.arctan2(opticAxis2[1], opticAxis2[0])
            lateralShiftX = np.cos(realPolarAngle) * lateralShift
            lateralShiftY = np.sin(realPolarAngle) * lateralShift
        elif np.abs(opticAxis1[2]) < np.abs(opticAxis2[2]):
            realPolarAngle = np.arctan2(opticAxis1[1], opticAxis1[0])
            lateralShiftX = np.cos(realPolarAngle) * lateralShift
            lateralShiftY = np.sin(realPolarAngle) * lateralShift
        else: 
            realPolarAngle = 0.0
            lateralShiftX = 0.0
            lateralShiftY = 0.0
        
        return realPolarAngle, lateralShiftX, lateralShiftY
