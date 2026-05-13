# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

from .FieldContainers import Domains, FieldContainer, RegularTransversalField, RegularElectricField, MeshTransversalField, MeshElectricField

class FourierTransforms: 
    @staticmethod
    def fft(x, y, field) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError('The x coordinates should be delivered in the form of a numpy array.')
        if not isinstance(y, np.ndarray):
            raise TypeError('The y coordinates should be delivered in the form of a numpy array.')
        if not isinstance(field, np.ndarray):
            raise TypeError('The field should be delivered in the form of a numpy array.')
        if np.shape(x) != np.shape(y) or np.shape(x) != np.shape(field):
            raise ValueError('The coordinates and the field should all have the same dimensions.')
        if not np.issubdtype(x.dtype, float): 
            raise TypeError('The x coordinates should be floats.')
        if not np.issubdtype(y.dtype, float):
            raise TypeError('The y coordinates should be floats.')
        if not (np.issubdtype(field.dtype, float) or np.issubdtype(field.dtype, complex)):
            raise TypeError('The field should be real- or complex-valued.')
        
        windowx = np.nanmax(x) - np.nanmin(x)
        windowy = np.nanmax(y) - np.nanmin(y)
        nx = np.shape(x)[0]
        ny = np.shape(x)[1]
        dx = windowx / (nx-1)
        dy = windowy / (ny-1)

        windowkx = 2.0 * np.pi / dx
        windowky = 2.0 * np.pi / dy
        kxmin = -windowkx / 2.0
        kxmax = windowkx / 2.0
        kymin = -windowky / 2.0
        kymax = windowky / 2.0
        kx = np.linspace(kxmin, kxmax, nx)
        ky = np.linspace(kymin, kymax, ny)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        jx = np.linspace(0, nx-1, nx)
        jy = np.linspace(0, ny-1, ny)
        jx, jy = np.meshgrid(jx, jy, indexing='ij')

        corr_exp_x = np.exp(1.0j * np.pi * (jx - kx/windowkx))
        corr_exp_y = np.exp(1.0j * np.pi * (jy - ky/windowky))

        spectrum = (0.5 * dx * dy / np.pi) * np.fft.fftshift(np.fft.fft2(field)) * corr_exp_x * corr_exp_y
        return kx, ky, spectrum
    
    @staticmethod
    def ifft(kx, ky, spectrum):
        if not isinstance(kx, np.ndarray):
            raise TypeError('The kx coordinates should be delivered in the form of a numpy array.')
        if not isinstance(ky, np.ndarray):
            raise TypeError('The ky coordinates should be delivered in the form of a numpy array.')
        if not isinstance(spectrum, np.ndarray):
            raise TypeError('The spectrum should be delivered in the form of a numpy array.')
        if np.shape(kx) != np.shape(ky) or np.shape(kx) != np.shape(spectrum):
            raise ValueError('The coordinates and the spectrum should all have the same dimensions.')
        if not np.issubdtype(kx.dtype, float): 
            raise TypeError('The kx coordinates should be floats.')
        if not np.issubdtype(ky.dtype, float):
            raise TypeError('The ky coordinates should be floats.')
        if not (np.issubdtype(spectrum.dtype, float) or np.issubdtype(spectrum.dtype, complex)):
            raise TypeError('The field should be real- or complex-valued.')

        if not isinstance(spectrum, RegularTransversalField):
            raise TypeError('The spectrum should be sampled on a regular Cartesian grid.')
        if spectrum.Domain != Domains.K:
            raise ValueError('The spectrum should be defined in the spatial frequency domain.')
        
        windowkx = np.nanmax(kx) - np.nanmin(kx)
        windowky = np.nanmax(ky) - np.nanmin(ky)
        nx = np.shape(kx)[0]
        ny = np.shape(kx)[1]
        dkx = windowkx / (nx-1)
        dky = windowky / (ny-1)

        windowx = 2.0 * np.pi / dkx
        windowy = 2.0 * np.pi / dky
        xmin = -windowx / 2.0
        xmax = windowx / 2.0
        ymin = -windowy / 2.0
        ymax = windowy / 2.0
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        x, y = np.meshgrid(x, y, indexing='ij')

        ix = np.linspace(0, nx-1, nx)
        iy = np.linspace(0, ny-1, ny)
        ix, iy = np.meshgrid(ix, iy, indexing='ij')

        corr_exp_x = np.exp(-1.0j * np.pi * (ix - x/windowx))
        corr_exp_y = np.exp(-1.0j * np.pi * (iy - y/windowy))

        field = (0.5 * dkx * dky * nx * ny / np.pi) * np.fft.fftshift(np.fft.ifft2(spectrum)) * corr_exp_x * corr_exp_y

        return x, y, field

    @staticmethod
    def FFT(field: RegularTransversalField) -> RegularTransversalField:
        if not isinstance(field, RegularTransversalField):
            raise TypeError('The field should be sampled on regular Cartesian grid.')
        if field.Domain != Domains.X:
            raise ValueError('The field should be defined in the space domain.')
        
        windowx = np.nanmax(field.X) - np.nanmin(field.X)
        windowy = np.nanmax(field.Y) - np.nanmin(field.Y)
        nx = np.shape(field.X)[0]
        ny = np.shape(field.X)[1]
        dx = windowx / (nx-1)
        dy = windowy / (ny-1)

        windowkx = 2.0 * np.pi / dx
        windowky = 2.0 * np.pi / dy
        kxmin = -windowkx / 2.0
        kxmax = windowkx / 2.0
        kymin = -windowky / 2.0
        kymax = windowky / 2.0
        kx = np.linspace(kxmin, kxmax, nx)
        ky = np.linspace(kymin, kymax, ny)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        jx = np.linspace(0, nx-1, nx)
        jy = np.linspace(0, ny-1, ny)
        jx, jy = np.meshgrid(jx, jy, indexing='ij')

        corr_exp_x = np.exp(1.0j * np.pi * (jx - kx/windowkx))
        corr_exp_y = np.exp(1.0j * np.pi * (jy - ky/windowky))

        spectrumX = (0.5 * dx * dy / np.pi) * np.fft.fftshift(np.fft.fft2(field.FieldX)) * corr_exp_x * corr_exp_y
        spectrumY = (0.5 * dx * dy / np.pi) * np.fft.fftshift(np.fft.fft2(field.FieldY)) * corr_exp_x * corr_exp_y
        return RegularTransversalField(field.Wavelength, field.Dispersion, kx, ky, spectrumX, spectrumY, Domains.K)
    
    @staticmethod
    def IFFT(spectrum: RegularTransversalField) -> RegularTransversalField:
        if not isinstance(spectrum, RegularTransversalField):
            raise TypeError('The spectrum should be sampled on a regular Cartesian grid.')
        if spectrum.Domain != Domains.K:
            raise ValueError('The spectrum should be defined in the spatial frequency domain.')
        
        windowkx = np.nanmax(spectrum.X) - np.nanmin(spectrum.X)
        windowky = np.nanmax(spectrum.Y) - np.nanmin(spectrum.Y)
        nx = np.shape(spectrum.X)[0]
        ny = np.shape(spectrum.X)[1]
        dkx = windowkx / (nx-1)
        dky = windowky / (ny-1)

        windowx = 2.0 * np.pi / dkx
        windowy = 2.0 * np.pi / dky
        xmin = -windowx / 2.0
        xmax = windowx / 2.0
        ymin = -windowy / 2.0
        ymax = windowy / 2.0
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        x, y = np.meshgrid(x, y, indexing='ij')

        ix = np.linspace(0, nx-1, nx)
        iy = np.linspace(0, ny-1, ny)
        ix, iy = np.meshgrid(ix, iy, indexing='ij')

        corr_exp_x = np.exp(-1.0j * np.pi * (ix - x/windowx))
        corr_exp_y = np.exp(-1.0j * np.pi * (iy - y/windowy))

        fieldX = (0.5 * dkx * dky * nx * ny / np.pi) * np.fft.fftshift(np.fft.ifft2(spectrum.FieldX)) * corr_exp_x * corr_exp_y
        fieldY = (0.5 * dkx * dky * nx * ny / np.pi) * np.fft.fftshift(np.fft.ifft2(spectrum.FieldY)) * corr_exp_x * corr_exp_y

        return RegularTransversalField(spectrum.Wavelength, spectrum.Dispersion, x, y, fieldX, fieldY, Domains.X)
    
    @staticmethod
    def FFT3(field: RegularElectricField) -> RegularElectricField:
        if not isinstance(field, RegularElectricField):
            raise TypeError('The field should be sampled on regular Cartesian grid.')
        if field.Domain != Domains.X:
            raise ValueError('The field should be defined in the space domain.')
        
        windowx = np.nanmax(field.X) - np.nanmin(field.X)
        windowy = np.nanmax(field.Y) - np.nanmin(field.Y)
        nx = np.shape(field.X)[0]
        ny = np.shape(field.X)[1]
        dx = windowx / (nx-1)
        dy = windowy / (ny-1)

        windowkx = 2.0 * np.pi / dx
        windowky = 2.0 * np.pi / dy
        kxmin = -windowkx / 2.0
        kxmax = windowkx / 2.0
        kymin = -windowky / 2.0
        kymax = windowky / 2.0
        kx = np.linspace(kxmin, kxmax, nx)
        ky = np.linspace(kymin, kymax, ny)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        jx = np.linspace(0, nx-1, nx)
        jy = np.linspace(0, ny-1, ny)
        jx, jy = np.meshgrid(jx, jy, indexing='ij')

        corr_exp_x = np.exp(1.0j * np.pi * (jx - kx/windowkx))
        corr_exp_y = np.exp(1.0j * np.pi * (jy - ky/windowky))

        spectrumX = (0.5 * dx * dy / np.pi) * np.fft.fftshift(np.fft.fft2(field.FieldX)) * corr_exp_x * corr_exp_y
        spectrumY = (0.5 * dx * dy / np.pi) * np.fft.fftshift(np.fft.fft2(field.FieldY)) * corr_exp_x * corr_exp_y
        spectrumZ = (0.5 * dx * dy / np.pi) * np.fft.fftshift(np.fft.fft2(field.FieldZ)) * corr_exp_x * corr_exp_y
        return RegularElectricField(field.Wavelength, field.Dispersion, kx, ky, spectrumX, spectrumY, spectrumZ, Domains.K)
    
    @staticmethod
    def IFFT3(spectrum: RegularElectricField) -> RegularElectricField:
        if not isinstance(spectrum, RegularElectricField):
            raise TypeError('The spectrum should be sampled on a regular Cartesian grid.')
        if spectrum.Domain != Domains.K:
            raise ValueError('The spectrum should be defined in the spatial frequency domain.')
        
        windowkx = np.nanmax(spectrum.X) - np.nanmin(spectrum.X)
        windowky = np.nanmax(spectrum.Y) - np.nanmin(spectrum.Y)
        nx = np.shape(spectrum.X)[0]
        ny = np.shape(spectrum.X)[1]
        dkx = windowkx / (nx-1)
        dky = windowky / (ny-1)

        windowx = 2.0 * np.pi / dkx
        windowy = 2.0 * np.pi / dky
        xmin = -windowx / 2.0
        xmax = windowx / 2.0
        ymin = -windowy / 2.0
        ymax = windowy / 2.0
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        x, y = np.meshgrid(x, y, indexing='ij')

        ix = np.linspace(0, nx-1, nx)
        iy = np.linspace(0, ny-1, ny)
        ix, iy = np.meshgrid(ix, iy, indexing='ij')

        corr_exp_x = np.exp(-1.0j * np.pi * (ix - x/windowx))
        corr_exp_y = np.exp(-1.0j * np.pi * (iy - y/windowy))

        fieldX = (0.5 * dkx * dky * nx * ny / np.pi) * np.fft.fftshift(np.fft.ifft2(spectrum.FieldX)) * corr_exp_x * corr_exp_y
        fieldY = (0.5 * dkx * dky * nx * ny / np.pi) * np.fft.fftshift(np.fft.ifft2(spectrum.FieldY)) * corr_exp_x * corr_exp_y
        fieldZ = (0.5 * dkx * dky * nx * ny / np.pi) * np.fft.fftshift(np.fft.ifft2(spectrum.FieldZ)) * corr_exp_x * corr_exp_y
        return RegularElectricField(spectrum.Wavelength, spectrum.Dispersion, x, y, fieldX, fieldY, fieldZ, Domains.X)
    
    @staticmethod
    def PFT(field: FieldContainer) -> MeshTransversalField:
        if not isinstance(field, MeshTransversalField):
            raise TypeError(f'Error in {FourierTransforms.__name__}.{FourierTransforms.PFT.__name__} → The field on which to apply this implementation of the Pointwise Fourier Transform must be an instance of the {MeshTransversalField.__name__} class.')
        if field.Domain != Domains.X:
            raise ValueError(f'Error in {FourierTransforms.__name__}.{FourierTransforms.PFT.__name__} → The field should be defined in the space domain.')
        
        triangulation = tri.Triangulation(field.X, field.Y)
        interpolator = tri.CubicTriInterpolator(triangulation, field.Wavefront)
        kxmesh = interpolator.gradient(field.X, field.Y)[0]
        kymesh = interpolator.gradient(field.X, field.Y)[1]
        interpolatorkx = tri.CubicTriInterpolator(triangulation, kxmesh)
        interpolatorky = tri.CubicTriInterpolator(triangulation, kymesh)
        hessian_xx = interpolatorkx.gradient(field.X, field.Y)[0]
        hessian_xy = interpolatorkx.gradient(field.X, field.Y)[1]
        hessian_yx = interpolatorky.gradient(field.X, field.Y)[0]
        hessian_yy = interpolatorky.gradient(field.X, field.Y)[1]

        hessian_det = np.array([hessian_xy*hessian_yx - hessian_xx*hessian_yy], dtype=complex)
        # alpha = 1.0 / np.sqrt((hessian_det))
        # alpha = np.where(
        #     np.sqrt(hessian_det) == 0.0, 
        #     np.ones(np.shape(hessian_det)), 
        #     1.0 / np.sqrt((hessian_det)))

        hessian_det[hessian_det==complex(0.0)] = 1.0
        alpha = 1/np.sqrt(hessian_det, dtype=complex)

        wavefrontk = field.Wavefront - kxmesh*field.X - kymesh*field.Y

        return MeshTransversalField(
            field.Wavelength, 
            field.Dispersion, 
            kxmesh, kymesh, 
            np.squeeze((alpha * field.FieldX).T), 
            np.squeeze((alpha * field.FieldY).T), 
            np.squeeze(wavefrontk.T), 
            Domains.K)
    
    @staticmethod
    def IPFT(spectrum: FieldContainer) -> MeshTransversalField:
        if not isinstance(spectrum, MeshTransversalField):
            raise TypeError(f'Error in {FourierTransforms.__name__}.{FourierTransforms.IPFT.__name__} → The field on which to apply this implementation of the Pointwise Fourier Transform must be an instance of the {MeshTransversalField.__name__} class.')
        if spectrum.Domain != Domains.K:
            raise ValueError('The spectrum should be defined in the spatial frequency domain.')
        
        triangulation = tri.Triangulation(spectrum.X, spectrum.Y)
        interpolator = tri.CubicTriInterpolator(triangulation, spectrum.Wavefront)
        xmesh = -interpolator.gradient(spectrum.X, spectrum.Y)[0]
        ymesh = -interpolator.gradient(spectrum.X, spectrum.Y)[1]
        interpolatorx = tri.CubicTriInterpolator(triangulation, xmesh)
        interpolatory = tri.CubicTriInterpolator(triangulation, ymesh)
        hessian_xx = interpolatorx.gradient(spectrum.X, spectrum.Y)[0]
        hessian_xy = interpolatorx.gradient(spectrum.X, spectrum.Y)[1]
        hessian_yx = interpolatory.gradient(spectrum.X, spectrum.Y)[0]
        hessian_yy = interpolatory.gradient(spectrum.X, spectrum.Y)[1]

        hessian_det = np.array([hessian_xy*hessian_yx - hessian_xx*hessian_yy], dtype=complex)
        # alpha = 1.0 / np.sqrt((hessian_det))
        # alpha = np.where(hessian_det==0.0, np.ones(np.shape(hessian_det)), alpha)
        hessian_det[hessian_det==complex(0.0)] = 1.0
        alpha = 1/np.sqrt(hessian_det, dtype=complex)

        wavefront = spectrum.Wavefront + spectrum.X * xmesh + spectrum.Y * ymesh

        return MeshTransversalField(
            spectrum.Wavelength, 
            spectrum.Dispersion, 
            xmesh, ymesh, 
            np.squeeze((alpha * spectrum.FieldX).T), 
            np.squeeze((alpha * spectrum.FieldY).T), 
            np.squeeze(wavefront.T), 
            Domains.X)
    
    @staticmethod
    def PFT3(field: MeshElectricField) -> MeshElectricField:
        if not isinstance(field, MeshElectricField):
            raise TypeError('The field should be sampled on a point cloud.')
        if field.Domain != Domains.X:
            raise ValueError('The field should be defined in the space domain.')
        
        triangulation = tri.Triangulation(field.X, field.Y)
        interpolator = tri.CubicTriInterpolator(triangulation, field.Wavefront)
        kxmesh = interpolator.gradient(field.X, field.Y)[0]
        kymesh = interpolator.gradient(field.X, field.Y)[1]
        interpolatorkx = tri.CubicTriInterpolator(triangulation, kxmesh)
        interpolatorky = tri.CubicTriInterpolator(triangulation, kymesh)
        hessian_xx = interpolatorkx.gradient(field.X, field.Y)[0]
        hessian_xy = interpolatorkx.gradient(field.X, field.Y)[1]
        hessian_yx = interpolatorky.gradient(field.X, field.Y)[0]
        hessian_yy = interpolatorky.gradient(field.X, field.Y)[1]

        hessian_det = np.array([hessian_xy*hessian_yx - hessian_xx*hessian_yy], dtype=complex)
        # alpha = 1.0 / np.sqrt((hessian_det))
        # alpha = np.where(hessian_det==0.0, np.ones(np.shape(hessian_det)), alpha)
        hessian_det[hessian_det==complex(0.0)] = 1.0
        alpha = 1/np.sqrt(hessian_det, dtype=complex)

        wavefrontk = field.Wavefront - kxmesh*field.X - kymesh*field.Y

        return MeshElectricField(
            field.Wavelength, 
            field.Dispersion, 
            kxmesh, 
            kymesh, 
            np.squeeze((alpha * field.FieldX).T), 
            np.squeeze((alpha * field.FieldY).T), 
            np.squeeze((alpha * field.FieldZ).T), 
            np.squeeze(wavefrontk.T), 
            Domains.K)
    
    @staticmethod
    def IPFT3(spectrum: MeshElectricField) -> MeshElectricField:
        if not isinstance(spectrum, MeshElectricField):
            raise TypeError('The spectrum should be sampled on a point cloud.')
        if spectrum.Domain != Domains.K:
            raise ValueError('The spectrum should be defined in the spatial frequency domain.')
        
        triangulation = tri.Triangulation(spectrum.X, spectrum.Y)
        interpolator = tri.CubicTriInterpolator(triangulation, spectrum.Wavefront)
        xmesh = -interpolator.gradient(spectrum.X, spectrum.Y)[0]
        ymesh = -interpolator.gradient(spectrum.X, spectrum.Y)[1]
        interpolatorx = tri.CubicTriInterpolator(triangulation, xmesh)
        interpolatory = tri.CubicTriInterpolator(triangulation, ymesh)
        hessian_xx = interpolatorx.gradient(spectrum.X, spectrum.Y)[0]
        hessian_xy = interpolatorx.gradient(spectrum.X, spectrum.Y)[1]
        hessian_yx = interpolatory.gradient(spectrum.X, spectrum.Y)[0]
        hessian_yy = interpolatory.gradient(spectrum.X, spectrum.Y)[1]

        hessian_det = np.array([hessian_xy*hessian_yx - hessian_xx*hessian_yy], dtype=complex)
        # alpha = 1.0 / np.sqrt((hessian_det))
        # alpha = np.where(hessian_det==0.0, np.ones(np.shape(hessian_det)), alpha)
        hessian_det[hessian_det==complex(0.0)] = 1.0
        alpha = 1/np.sqrt(hessian_det, dtype=complex)

        wavefront = spectrum.Wavefront + spectrum.X * xmesh + spectrum.Y * ymesh

        return MeshElectricField(
            spectrum.Wavelength, 
            spectrum.Dispersion, 
            xmesh, 
            ymesh, 
            np.squeeze((alpha * spectrum.FieldX).T), 
            np.squeeze((alpha * spectrum.FieldY).T), 
            np.squeeze((alpha * spectrum.FieldZ).T), 
            np.squeeze(wavefront.T), 
            Domains.X)