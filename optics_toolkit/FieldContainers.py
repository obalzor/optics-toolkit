# SPDX-FileCopyrightText: 2026 Olga Baladron-Zorita
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colours
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from enum import Enum
from abc import ABC, abstractmethod
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator, griddata

from .Constants import Constants
from .Materials import Material
from .Miscellanea import Miscellanea, Defaults

class Domains(Enum): 
    X = 1
    K = 2

class FieldContainer(ABC): 
    @property
    @abstractmethod
    def Wavelength(self) -> float: 
        pass

    @property
    @abstractmethod
    def Dispersion(self) -> Material: 
        pass

    @property
    @abstractmethod
    def X(self) -> np.ndarray: 
        pass

    @property
    @abstractmethod
    def Y(self) -> np.ndarray: 
        pass

    @property
    @abstractmethod
    def FieldX(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def FieldY(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def Domain(self) -> Domains: 
        pass

    @abstractmethod
    def Display(self): 
        pass

class RegularTransversalField(FieldContainer): 
    def __init__(
            self, 
            wavelength: float, 
            dispersion: Material, 
            x: np.ndarray, 
            y: np.ndarray, 
            fieldX: np.ndarray, 
            fieldY: np.ndarray, 
            domain: Domains
    ): 
        if not isinstance(wavelength, float): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The wavelength must be a float.')
        if wavelength <= 0.0: 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The wavelength must be strictly positive.')
        if not isinstance(dispersion, Material): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The dispersion characteristics of the medium in which the field is defined must be provided in the form of an instance of a class derived from Material.')
        if not (dispersion.IsWavelengthWithinRange(wavelength)): 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The wavelength must be within the allowed range by the dispersion characteristics of the medium in which the field is defined.')
        if not isinstance(x, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The set of x coordinates must be a numpy array.')
        if not np.issubdtype(x.dtype, float): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The dtype of the set of x coordinates must be float.')
        if not isinstance(y, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The set of y coordinates must be a numpy array.')
        if not np.issubdtype(y.dtype, float): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The dtype of the set of y coordinates must be float.')
        if np.shape(y) != np.shape(x): 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The dimensions of the set of x and y coordinates must be the same.')
        if not isinstance(fieldX, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The set of Ex values must be a numpy array.')
        if not np.issubdtype(fieldX.dtype, complex): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The Ex entries must be complex.')
        if np.shape(fieldX) != np.shape(x): 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The dimensions of the set of Ex values must be the same as those of the x and y coordinates.')
        if not isinstance(fieldY, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The set of Ex values must be a numpy array.')
        if not np.issubdtype(fieldY.dtype, complex): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The Ey entries must be complex.')
        if np.shape(fieldY) != np.shape(x): 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The dimensions of the set of Ey values must be the same as those of the x and y coordinates.')
        if not isinstance(domain, Domains): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.__init__.__name__} → The domain in which the field is defined must belong to the enumerator Domains.')
        
        self._wavelength = wavelength
        self._dispersion = dispersion
        self._x = x
        self._y = y
        self._fieldX = fieldX
        self._fieldY = fieldY
        self._domain = domain
    
    @property
    def Dispersion(self): 
        return self._dispersion
    @Dispersion.setter
    def Dispersion(self, value): 
        if not isinstance(value, Material): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Dispersion.__name__} → The dispersion properties of the material in which the field is defined must be delivered as an instance of a class derived from {Material.__name__}.')
        self._dispersion = value

    @property
    def Wavelength(self): 
        return self._wavelength
    @Wavelength.setter
    def Wavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Wavelength.__name__} → The wavelength must be a float.')
        if value <= 0.0: 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Wavelength.__name__} → The wavelength must be positive.')
        if not self.Dispersion.IsWavelengthWithinRange(value): 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Wavelength.__name__} → The wavelength must be within the allowed range by the dispersion characteristics of the medium in which the field is defined.')
        self._wavelength = value
    
    @property
    def X(self): 
        return self._x
    @X.setter
    def X(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.X.__name__} → The set of x coordinates must be delivered as a numpy array. ')
        if not np.issubdtype(value.dtype, float): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.X.__name__} → The set of x coordinates must have float as its dtype.')
        self._x = value

    @property
    def Y(self): 
        return self._y
    @Y.setter
    def Y(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Y.__name__} → The set of y coordinates must be delivered as a numpy array.')
        if not np.issubdtype(value.dtype, float): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Y.__name__} → The set of y coordinates must have float as its dtype.')
        self._y = value

    @property
    def FieldX(self): 
        return self._fieldX
    @FieldX.setter
    def FieldX(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.FieldX.__name__} → The set of Ex field values must be delivered as a numpy array.')
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.FieldX.__name__} → The set of Ex field values must have float as its dtype.')
        self._fieldX = value
    
    @property
    def FieldY(self): 
        return self._fieldY
    @FieldY.setter
    def FieldY(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.FieldY.__name__} → The set of Ey field values must be delivered as a numpy array.')
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.FieldY.__name__} → The set of Ey field valus must have float as its dtype.')
        self._fieldY = value

    @property
    def Domain(self): 
        return self._domain
    @Domain.setter
    def Domain(self, value): 
        if not isinstance(value, Domains): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Domain.__name__} → The domain in which the field is defined must be indicated using the Enum {Domains.__name__}.')
        self._domain = value
    
    @property
    def Extent(self): 
        return np.array([np.nanmin(self.X), np.nanmax(self.X), np.nanmin(self.Y), np.nanmax(self.Y)])
    
    @property
    def WindowX(self): 
        return np.nanmax(self.X) - np.nanmin(self.X)
    
    @property
    def WindowY(self): 
        return np.nanmax(self.Y) - np.nanmin(self.Y)
    
    @property
    def PixelsX(self): 
        return np.shape(self.X)[0]
    
    @property
    def PixelsY(self): 
        return np.shape(self.X)[1]
    
    @property
    def IndexOfCentralPixelX(self): 
        return int(self.PixelsX/2)
    
    @property
    def IndexOfCentralPixelY(self): 
        return int(self.PixelsY/2)
    
    @property
    def CoordinateOfCentralPixelX(self): 
        return self.X[self.IndexOfCentralPixelX, self.IndexOfCentralPixelY]
    
    @property
    def CoordinateOfCentralPixelY(self): 
        return self.Y[self.IndexOfCentralPixelX, self.IndexOfCentralPixelY]
    
    @property
    def XMin(self): 
        return np.nanmin(self.X)
    
    @property
    def XMax(self): 
        return np.nanmax(self.X)
    
    @property
    def YMin(self): 
        return np.nanmin(self.Y)
    
    @property
    def YMax(self): 
        return np.nanmax(self.Y)
    
    @property
    def PitchX(self): 
        return self.X[1,0] - self.X[0,0]
    
    @property
    def PitchY(self): 
        return self.Y[0,1] - self.Y[0,0]
    
    @property
    def K0(self): 
        return 2.0 * np.pi / self.Wavelength
    
    @property
    def AngularFrequency(self): 
        return self.K0 * Constants.C
    
    @property
    def Wavenumber(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetValue(self.Wavelength)
    
    @property
    def WavenumberReal(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength)
    
    @property
    def Kz(self): 
        if self.Domain != Domains.K: 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.Kz.__name__} → The field must be defined in the K domain in order to calculate the Kz component of the wave-vector.')
        kz = np.zeros_like(self.X)
        kz = np.sqrt(
            np.power(self.Wavenumber, 2) - 
            np.power(self.X, 2) - 
            np.power(self.Y, 2), dtype=complex)
        kz[np.isnan(kz)] = 0.0
        return kz
    
    @property
    def RealKz(self): 
        if self.Domain != Domains.K: 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.RealKz.__name__} → The field must be defined in the K domain in order to calculate the Kz component of the wave-vector.')
        return np.real(self.Kz)
    
    @property
    def ParaxialIntensity(self): 
        return np.power(np.abs(self.FieldX), 2) + np.power(np.abs(self.FieldY), 2)
    
    def CentredExtraction(self, windowFactor): 
        if not (isinstance(windowFactor, float) or isinstance(windowFactor, int)): 
            raise TypeError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.CentredExtraction.__name__} → The window factor for the extraction must be delivered as an int or float.')
        if windowFactor <= 0.0: 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.CentredExtraction.__name__} → The window factor must be strictly positive.')
        if windowFactor > 1.0: 
            raise ValueError(f'Error in {RegularTransversalField.__name__}.{RegularTransversalField.CentredExtraction.__name__} → With a window factor larger than 1 the operation you want is not an extraction. You want to embed the field instead.')
        
        # Calculate number of pixels in zoomed field
        nx = int(self.PixelsX * windowFactor)
        ny = int(self.PixelsY * windowFactor)

        # Ensure that number of pixels is odd
        if nx%2 == 0: 
            nx += 1
        if ny%2 == 0: 
            ny += 1
        
        # Create containers for zoomed field
        xZoom = np.zeros((nx, ny), dtype=float)
        yZoom = np.zeros((nx, ny), dtype=float)
        fieldXZoom = np.zeros((nx, ny), dtype=complex)
        fieldYZoom = np.zeros((nx, ny), dtype=complex)

        if windowFactor == 1.0: 
            return self
        else: 
            # Calculate indices of range
            startPixelX = self.IndexOfCentralPixelX - int(nx/2)
            endPixelX = self.IndexOfCentralPixelX + int(nx/2) + 1
            startPixelY = self.IndexOfCentralPixelY - int(ny/2)
            endPixelY = self.IndexOfCentralPixelY + int(ny/2) + 1
            xZoom = self.X[startPixelX:endPixelX, startPixelY:endPixelY]
            yZoom = self.Y[startPixelX:endPixelX, startPixelY:endPixelY]
            fieldXZoom = self.FieldX[startPixelX:endPixelX, startPixelY:endPixelY]
            fieldYZoom = self.FieldY[startPixelX:endPixelX, startPixelY:endPixelY]
        
        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            xZoom, yZoom, 
            fieldXZoom, fieldYZoom, 
            self.Domain)
    
    def CentredEmbedding(self, windowFactor): 
        if not (isinstance(windowFactor, int) or isinstance(windowFactor, float)): 
            raise TypeError('The window factor for the embedding must be delivered as an int or float.')
        if windowFactor < 1.0: 
            raise ValueError('For a window factor between 0 and 1, use extraction instead.')
        
        # Calculate number of pixels in embedded field
        windowx = windowFactor * self.WindowX
        windowy = windowFactor * self.WindowY

        nx = int(round(windowx / self.PitchX) + 1)
        ny = int(round(windowy / self.PitchY) + 1)

        # Ensure that number of pixels is odd
        if nx%2 == 0: 
            nx += 1
        if ny%2 == 0: 
            ny += 1
        
        # Recalculate window size 
        windowx = (nx-1) * self.PitchX
        windowy = (ny-1) * self.PitchY
        
        # Create containers for embedded field
        xEmbed = np.zeros((nx, ny), dtype=float)
        yEmbed = np.zeros((nx, ny), dtype=float)
        fieldXEmbed = np.zeros((nx, ny), dtype=complex)
        fieldYEmbed = np.zeros((nx, ny), dtype=complex)

        # Create coordinates
        xEmbed = np.linspace(-0.5*windowx, 0.5*windowx, nx)
        yEmbed = np.linspace(-0.5*windowy, 0.5*windowy, ny)
        xEmbed, yEmbed = np.meshgrid(xEmbed, yEmbed, indexing='ij')

        # Embed field
        startPixelX = int(nx/2) - int(self.PixelsX/2)
        endPixelX = int(nx/2) + int(self.PixelsX/2) + 1
        startPixelY = int(ny/2) - int(self.PixelsY/2)
        endPixelY = int(ny/2) + int(self.PixelsY/2) + 1

        fieldXEmbed[startPixelX:endPixelX, startPixelY:endPixelY] = self.FieldX
        fieldYEmbed[startPixelX:endPixelX, startPixelY:endPixelY] = self.FieldY

        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            xEmbed, yEmbed, 
            fieldXEmbed, fieldYEmbed, 
            self.Domain)
    
    def Display(self, **optionalArguments): 
        allowedOptionalArguments = {
            'figureSize', 
            'colourMapAmplitude', 
            'colourMapPhase', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks',
            'windowSizeFactor',
            'normaliseColourScale'
        }
        unrecognisedKeys = set(optionalArguments) - allowedOptionalArguments
        if unrecognisedKeys:
            raise TypeError(f'Unexpected keyword arguments: {", ".join(unrecognisedKeys)}')

        # Retrieve optional arguments
        figureSize = optionalArguments.get('figureSize', Defaults.defaultFieldFigureSize)
        colourMapAmplitude = optionalArguments.get('colourMapAmplitude', Defaults.defaultAmplitudeColourMap)
        colourMapPhase = optionalArguments.get('colourMapPhase', Defaults.defaultPhaseColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        windowSizeFactor = optionalArguments.get('windowSizeFactor', 1.0)
        normaliseColourScale = optionalArguments.get('normaliseColourScale', True)

        normAmplitude = colours.Normalize(vmin=0.0, vmax=np.nanmax((np.abs(self.FieldX), np.abs(self.FieldY))))
        normPhase = colours.Normalize(vmin=-np.pi, vmax=np.pi)

        if windowSizeFactor == 1.0: 
            field = self
        elif windowSizeFactor > 1.0: 
            field = self.CentredEmbedding(windowSizeFactor)
        elif windowSizeFactor < 1.0: 
            field = self.CentredExtraction(windowSizeFactor)
        else: 
            raise ValueError('No other option possible')
        
        if self.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif self.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(self.X, self.Y)
        else: 
            raise ValueError('No other possibility.')

        fig, axs = plt.subplots(
            nrows=2, ncols=2, 
            figsize=figureSize,
            sharex=True, sharey=True)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        if normaliseColourScale:
            cax11 = axs[0][0].imshow(
                np.abs(field.FieldX).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                norm=normAmplitude, 
                origin='lower')
        else: 
            cax11 = axs[0][0].imshow(
                np.abs(field.FieldX).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                origin='lower')
        if normaliseColourScale:
            cax12 = axs[0][1].imshow(
                np.abs(field.FieldY).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                norm=normAmplitude, 
                origin='lower')
        else: 
            cax12 = axs[0][1].imshow(
                np.abs(field.FieldY).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                origin='lower')
        cax21 = axs[1][0].imshow(
            np.angle(field.FieldX).T, 
            extent=factorExtent*field.Extent, 
            cmap=colourMapPhase, 
            norm=normPhase, 
            origin='lower')
        cax22 = axs[1][1].imshow(
            np.angle(field.FieldY).T, 
            extent=factorExtent*field.Extent, 
            cmap=colourMapPhase, 
            norm=normPhase, 
            origin='lower')
        cbar11 = fig.colorbar(cax11)
        cbar12 = fig.colorbar(cax12)
        cbar21 = fig.colorbar(cax21)
        cbar22 = fig.colorbar(cax22)
        axs[0][0].set_title('$|E_x|$', fontname=font, fontsize=fontsizeTitle)
        # plt.xticks(fontsize=fontsizeTicks, fontname=font)
        # plt.yticks(fontsize=fontsizeTicks, fontname=font)
        axs[0][1].set_title('$|E_y|$', fontname=font, fontsize=fontsizeTitle)
        # plt.xticks(fontsize=fontsizeTicks, fontname=font)
        # plt.yticks(fontsize=fontsizeTicks, fontname=font)
        axs[1][0].set_title('arg($E_x$) [rad]', fontname=font, fontsize=fontsizeTitle)
        # plt.xticks(fontsize=fontsizeTicks, fontname=font)
        # plt.yticks(fontsize=fontsizeTicks, fontname=font)
        axs[1][1].set_title('arg($E_y$) [rad]', fontname=font, fontsize=fontsizeTitle)
        # plt.xticks(fontsize=fontsizeTicks, fontname=font)
        # plt.yticks(fontsize=fontsizeTicks, fontname=font)
        for ax in axs.flat: 
            ax.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
            ax.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        plt.tight_layout()
        del field
        return fig
    
    def DisplayParaxialIntensity(self, **optionalArguments): 
        allowedParameters = {
            'figureSize', 
            'colourMapIntensity', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks', 
            'windowSizeFactor'
        }
        unrecognisedArguments = set(optionalArguments) - allowedParameters
        if unrecognisedArguments:
            raise TypeError(f'Not recognised: {", ".join(unrecognisedArguments)}.')
        
        # Retrieve optional parameters
        figureSize = optionalArguments.get('figureSize', Defaults.defaultSquareFigureSize)
        colourMapIntensity = optionalArguments.get('colourMapIntensity', Defaults.defaultIntensityColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        windowSizeFactor = optionalArguments.get('windowSizeFactor', 1.0)

        if windowSizeFactor == 1.0: 
            field = self
        elif windowSizeFactor < 1.0: 
            field = self.CentredExtraction(windowSizeFactor)
        elif windowSizeFactor > 1.0: 
            field = self.CentredEmbedding(windowSizeFactor)
        else: 
            raise ValueError('Not possible!')

        if field.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif field.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(field.X, field.Y)
        
        normIntensity = colours.Normalize(vmin=0.0, vmax=np.nanmax(field.ParaxialIntensity))

        fig, main_ax = plt.subplots(figsize=figureSize)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        divider = make_axes_locatable(main_ax)
        top_ax = divider.append_axes('top', 2.0, pad=0.3, sharex=main_ax)
        right_ax = divider.append_axes('right', 2.0, pad=0.3, sharey=main_ax)

        top_ax.xaxis.set_tick_params(labelbottom=False)
        right_ax.yaxis.set_tick_params(labelleft=False)

        main_ax.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        main_ax.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        top_ax.set_ylabel('$\\sum |E_\\perp|^2$', fontname=font, fontsize=fontsizeAxes)
        right_ax.set_xlabel('$\\sum |E_\\perp|^2$', fontname=font, fontsize=fontsizeAxes)
        main_ax.imshow(field.ParaxialIntensity.T, cmap=colourMapIntensity, extent=factorExtent*np.array(field.Extent), norm=normIntensity, origin='lower')
        main_ax.autoscale(enable=False)
        right_ax.autoscale(enable=False)
        top_ax.autoscale(enable=False)
        right_ax.set_xlim(right=np.nanmax(field.ParaxialIntensity))
        top_ax.set_ylim(top=np.nanmax(field.ParaxialIntensity))
        v_line = main_ax.axvline(factorExtent*field.CoordinateOfCentralPixelX, color='white', linestyle='-.', linewidth=1)
        h_line = main_ax.axhline(factorExtent*field.CoordinateOfCentralPixelY, color='white', linestyle='-.', linewidth=1)
        v_prof = right_ax.plot(field.ParaxialIntensity[int(field.PixelsX/2),:], factorExtent*field.Y[int(field.PixelsX/2),:], color='black', linewidth=2)
        h_prof = top_ax.plot(factorExtent*field.X[:, int(field.PixelsY/2)], field.ParaxialIntensity[:, int(field.PixelsY/2)], color='black', linewidth=2)
        del field
        return fig
    
    def TransformToMesh(self, xmesh: np.ndarray, ymesh: np.ndarray): 
        if not isinstance(xmesh, np.ndarray): 
            raise TypeError('The x coordinates of the point cloud must be delivered as a numpy array.')
        if not isinstance(ymesh, np.ndarray): 
            raise TypeError('The y coordinates of the point cloud must be delivered as a numpy array.')
        if not np.issubdtype(xmesh.dtype, float): 
            raise TypeError('The dtype of the x coordinates of the point cloud must be float.')
        if not np.issubdtype(ymesh.dtype, float): 
            raise TypeError('The dtype of the y coordinates of the point cloud must be float.')
        
        # Check overlap between supports
        if np.nanmax(xmesh) <= self.XMin or np.nanmin(xmesh) >= self.XMax or np.nanmax(ymesh) <= self.YMin or np.nanmin(ymesh) >= self.YMax: 
            raise ValueError('There is no overlap between the supports.')
        
        # fieldX = griddata((self.X.ravel(), self.Y.ravel()), self.FieldX.ravel(), (xmesh, ymesh))
        # fieldY = griddata((self.X.ravel(), self.Y.ravel()), self.FieldY.ravel(), (xmesh, ymesh))

        x = self.X[:,0]
        y = self.Y[0,:]

        interpolatorFieldX = RegularGridInterpolator((x, y), self.FieldX, method='linear', bounds_error=False, fill_value=0.0)
        interpolatorFieldY = RegularGridInterpolator((x, y), self.FieldY, method='linear', bounds_error=False, fill_value=0.0)

        fieldX = interpolatorFieldX(np.column_stack((xmesh, ymesh)))
        fieldY = interpolatorFieldY(np.column_stack((xmesh, ymesh)))

        wavefront = np.zeros_like(xmesh)

        return MeshTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            xmesh, ymesh, 
            fieldX, fieldY, wavefront, 
            self.Domain)
    
    def Propagate(self, propagationDistance: float): 
        if not (isinstance(propagationDistance, int) or isinstance(propagationDistance, float)): 
            raise TypeError('The propagation distance must be an int or float.')
        if self.Domain != Domains.K: 
            raise ValueError('The field must be defined in the K domain before being propagated.')
        fieldX = self.FieldX * np.exp(1.0j * self.Kz * propagationDistance)
        fieldY = self.FieldY * np.exp(1.0j * self.Kz * propagationDistance)
        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, self.Y, 
            fieldX, fieldY, 
            self.Domain)
    
    def ApplyLateralShift(self, lateralShiftX: float, lateralShiftY: float): 
        if not isinstance(lateralShiftX, float): 
            raise TypeError('The lateralShiftX must be a float.')
        if not isinstance(lateralShiftY, float): 
            raise TypeError('The lateralShiftY must be a float.')
        
        if self.Domain == Domains.K: 
            fieldX = self.FieldX * np.exp(1.0j * (lateralShiftX * self.X + lateralShiftY * self.Y))
            fieldY = self.FieldY * np.exp(1.0j * (lateralShiftX * self.X + lateralShiftY * self.Y))
            return RegularTransversalField(
                self.Wavelength, 
                self.Dispersion, 
                self.X, self.Y, 
                fieldX, fieldY, 
                self.Domain)
        else: 
            raise ValueError('Only K domain implemented so far.')
    
    def CalculateCircularPolarisation(self):
        fieldL = (self.FieldX + 1.0j * self.FieldY) / np.sqrt(2)
        fieldR = (self.FieldX - 1.0j * self.FieldY) / np.sqrt(2)
        return RegularTransversalField(
            wavelength = self.Wavelength, 
            dispersion = self.Dispersion, 
            x = self.X, 
            y = self.Y, 
            fieldX = fieldL, 
            fieldY = fieldR, 
            domain = self.Domain)
    
    def CalculateEz(self): 
        if self.Domain != Domains.K: 
            raise ValueError(f'Error in {RegularTransversalField}.{RegularTransversalField.CalculateEz.__name__} → The Ez component can only be calculated in the K domain.')
        
        ez = -(self.X * self.FieldX + self.Y * self.FieldY) / self.Kz
        ez[np.isnan(ez)] = 0.0
        
        return RegularElectricField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, self.Y, 
            self.FieldX, self.FieldY, ez, 
            self.Domain)

class MeshTransversalField(FieldContainer): 
    def __init__(
            self, 
            wavelength: float, 
            dispersion: Material, 
            x: np.ndarray, 
            y: np.ndarray, 
            fieldX: np.ndarray, 
            fieldY: np.ndarray, 
            wavefront: np.ndarray, 
            domain: Domains
    ): 
        if not isinstance(wavelength, float): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The wavelength must be a float.')
        if wavelength <= 0.0: 
            raise ValueError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The wavelength must be positive.')
        if not isinstance(dispersion, Material): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dispersion characteristics of the medium in which the field is defined must be delivered as an instance of a class derived from {Material.__name__}.')
        if not dispersion.IsWavelengthWithinRange(wavelength): 
            raise ValueError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The wavelength must be within the range allowed by the dispersion characteristics of the material in which the field is defined.')
        if not isinstance(x, np.ndarray): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The set of x coordinates must be delivered as a numpy array.')
        if not np.issubdtype(x.dtype, float):
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dtype of the set of x coordinates must be float.')
        if not isinstance(y, np.ndarray): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The set of y coordinates must be delivered as a numpy array.')
        if not np.issubdtype(y.dtype, float): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dtype of the set of y coordinates must be float.')
        if np.shape(y) != np.shape(x): 
            raise ValueError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dimensions of the sets of x and y coordinates must be the same.')
        if not isinstance(fieldX, np.ndarray): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The set of Ex field values must be delivered as a numpy array.')
        if not np.issubdtype(fieldX.dtype, complex): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dtype of the set of Ex field values must be complex.')
        if np.shape(fieldX) != np.shape(x): 
            raise ValueError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dimensions of the set of Ex field values and the dimensions of the coordinates must be the same.')
        if not isinstance(fieldY, np.ndarray): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The set of Ey field values must be delivered as a numpy array.')
        if not np.issubdtype(fieldY.dtype, complex): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dtype of the set of Ey field values must be complex.')
        if np.shape(fieldY) != np.shape(x): 
            raise ValueError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dimensions of the set of Ey field balues and the dimensions of the coordinates must be the same.')
        if not isinstance(wavefront, np.ndarray):
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The set of wavefront values must be delivered as a numpy array.')
        if not np.issubdtype(wavefront.dtype, float): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dtype of the set of wavefront values must be float.')
        if np.shape(wavefront) != np.shape(x): 
            raise ValueError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The dimensions of the set of wavefront values and the dimensions of the coordinates must be the same.')
        if not isinstance(domain, Domains): 
            raise TypeError(f'Error in {MeshTransversalField.__name__}.{MeshTransversalField.__init__.__name__} → The domain in which the field is defined must be indicated through the enum {Domains.__name__}.')
        self._wavelength = wavelength
        self._dispersion = dispersion
        self._x = x
        self._y = y
        self._fieldX = fieldX
        self._fieldY = fieldY
        self._wavefront = wavefront
        self._domain = domain
    
    @property
    def Dispersion(self): 
        return self._dispersion
    @Dispersion.setter
    def Dispersion(self, value): 
        if not isinstance(value, Material): 
            raise TypeError('The dispersion properties of the material in which the field is defined must be delivered as an instance of a class derived from Material.')
        self._dispersion = value
    
    @property
    def Wavelength(self): 
        return self._wavelength
    @Wavelength.setter
    def Wavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The wavelength must be a float.')
        if value <= 0.0: 
            raise ValueError('The wavelength must be strictly positive.')
        if not self.Dispersion.IsWavelengthWithinRange(value): 
            raise ValueError('The wavelength must be within the allowed range.')
        self._wavelength = value
    
    @property
    def X(self): 
        return self._x
    @X.setter
    def X(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._x = value

    @property
    def Y(self): 
        return self._y
    @Y.setter
    def Y(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._y = value

    @property
    def FieldX(self): 
        return self._fieldX
    @FieldX.setter
    def FieldX(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldX = value
    
    @property
    def FieldY(self): 
        return self._fieldY
    @FieldY.setter
    def FieldY(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldY = value

    @property
    def Wavefront(self): 
        return self._wavefront
    @Wavefront.setter
    def Wavefront(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._wavefront = value
    
    @property
    def Domain(self): 
        return self._domain
    @Domain.setter
    def Domain(self, value): 
        if not isinstance(value, Domains): 
            raise TypeError()
        self._domain = value
    
    @property
    def Extent(self): 
        return np.array([np.nanmin(self.X), np.nanmax(self.X), np.nanmin(self.Y), np.nanmax(self.Y)])
    
    @property
    def WindowX(self): 
        return np.nanmax(self.X) - np.nanmin(self.X)
    
    @property
    def WindowY(self): 
        return np.nanmax(self.Y) - np.nanmin(self.Y)
    
    @property
    def Pixels(self): 
        return len(self.X)
    
    @property
    def CoordinateOfCentralPixelX(self): 
        return self.X[0]
    
    @property
    def CoordinateOfCentralPixelY(self): 
        return self.Y[0]
    
    @property
    def XMin(self): 
        return np.nanmin(self.X)
    
    @property
    def XMax(self): 
        return np.nanmax(self.X)
    
    @property
    def YMin(self): 
        return np.nanmin(self.Y)
    
    @property
    def YMax(self): 
        return np.nanmax(self.Y)
    
    @property
    def K0(self): 
        return 2.0 * np.pi / self.Wavelength
    
    @property
    def AngularFrequency(self): 
        return self.K0 * Constants.C
    
    @property
    def Wavenumber(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetValue(self.Wavelength)
    
    @property
    def WavenumberReal(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength)
    
    @property
    def Kz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('Kz can only be calculated when the field is defined in the K domain.')
        kz = np.zeros_like(self.X)
        kz = np.sqrt(
            np.power(self.Wavenumber, 2) - 
            np.power(self.X, 2) - 
            np.power(self.Y, 2), dtype=complex)
        kz[np.isnan(kz)] = 0.0
        return kz
    
    @property
    def RealKz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('Kz can only be calculated when the field is defined in the K domain.')
        return np.real(self.Kz)
    
    @property
    def ParaxialIntensity(self): 
        return np.power(np.abs(self.FieldX), 2) + np.power(np.abs(self.FieldY), 2)
    
    @property
    def WavefrontRange(self): 
        return np.nanmax(self.Wavefront) - np.nanmin(self.Wavefront)

    def Display(self, **optionalArguments): 
        allowedArguments = {
            'figureSize', 
            'colourMapAmplitude', 
            'colourMapPhase', 
            'colourMapWavefront', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks', 
            'normaliseColourScale'
        }
        unrecognisedArguments = set(optionalArguments) - allowedArguments
        if unrecognisedArguments: 
            raise TypeError(f'Unrecognised arguments: {", ".join(unrecognisedArguments)}.')
        
        figureSize = optionalArguments.get('figureSize', Defaults.defaultFieldFigureSize)
        colourMapAmplitude = optionalArguments.get('colourMapAmplitude', Defaults.defaultAmplitudeColourMap)
        colourMapPhase = optionalArguments.get('colourMapPhase', Defaults.defaultPhaseColourMap)
        colourMapWavefront = optionalArguments.get('colourMapWavefront', Defaults.defaultWavefrontColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        normaliseColourScale = optionalArguments.get('normaliseColourScale', True)

        normAmplitude = colours.Normalize(vmin=0.0, vmax=np.nanmax((np.abs(self.FieldX), np.abs(self.FieldY))))
        normPhase = colours.Normalize(vmin=-np.pi, vmax=np.pi)
        
        if self.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif self.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(self.X, self.Y)
        
        if self.Domain == Domains.K: 
            propagationCircle11 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle12 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle21 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle22 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircleWa = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)

        fig = plt.figure(figsize=figureSize)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        gs = GridSpec(nrows=2, ncols=4, figure=fig)
        ax11 = plt.subplot(gs.new_subplotspec((0,0)))
        ax12 = plt.subplot(gs.new_subplotspec((0,1)))
        ax21 = plt.subplot(gs.new_subplotspec((1,0)))
        ax22 = plt.subplot(gs.new_subplotspec((1,1)))
        axwa = plt.subplot(gs.new_subplotspec((0,2), colspan=2, rowspan=2))
        if normaliseColourScale:
            cax11 = ax11.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldX), cmap=colourMapAmplitude, norm=normAmplitude, s=10)
            cax12 = ax12.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldY), cmap=colourMapAmplitude, norm=normAmplitude, s=10)
        else: 
            cax11 = ax11.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldX), cmap=colourMapAmplitude, s=10)
            cax12 = ax12.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldY), cmap=colourMapAmplitude, s=10)
        cax21 = ax21.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.angle(self.FieldX), cmap=colourMapPhase, norm=normPhase, s=10)
        cax22 = ax22.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.angle(self.FieldY), cmap=colourMapPhase, norm=normPhase, s=10)
        caxwa = axwa.scatter(factorExtent*self.X, factorExtent*self.Y, c=self.Wavefront, cmap=colourMapWavefront, s=10)
        if self.Domain == Domains.K and (self.WindowX > 0.5*self.Wavenumber or self.WindowY > 0.5*self.Wavenumber):
            cax11Circle = ax11.add_patch(propagationCircle11)
            cax12Circle = ax12.add_patch(propagationCircle12)
            cax21Circle = ax21.add_patch(propagationCircle21)
            cax22Circle = ax22.add_patch(propagationCircle22)
            caxWaCircle = axwa.add_patch(propagationCircleWa)
        ax11.axis('equal')
        ax12.axis('equal')
        ax21.axis('equal')
        ax22.axis('equal')
        axwa.axis('equal')
        ax11.set_title('$|E_x|$', fontname=font, fontsize=fontsizeTitle)
        ax12.set_title('$|E_y|$', fontname=font, fontsize=fontsizeTitle)
        ax21.set_title('arg($E_x$) [rad]', fontname=font, fontsize=fontsizeTitle)
        ax22.set_title('arg($E_y$) [rad]', fontname=font, fontsize=fontsizeTitle)
        axwa.set_title('Wavefront [rad]', fontname=font, fontsize=fontsizeTitle)
        axwa.text(np.nanmin(factorExtent*self.X), np.nanmin(factorExtent*self.Y), f'{self.WavefrontRange:.2f} rad', fontname=font, fontsize=fontsizeAxes)
        cbar11 = fig.colorbar(cax11)
        cbar12 = fig.colorbar(cax12)
        cbar21 = fig.colorbar(cax21)
        cbar22 = fig.colorbar(cax22)
        cbarwa = fig.colorbar(caxwa)
        ax11.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        ax21.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        ax21.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        ax22.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        axwa.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        plt.tight_layout()
        return fig
    
    def Propagate(self, propagationDistance): 
        if not (isinstance(propagationDistance, int) or isinstance(propagationDistance, float)): 
            raise TypeError('The propagation distance must be delivered as an int or float.')
        if self.Domain != Domains.K: 
            raise ValueError('The field can only be propagated if it is defined in the K domain.')
        
        wavefront = self.Wavefront + np.real(self.Kz) * propagationDistance
        return MeshTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, self.Y, 
            self.FieldX, self.FieldY, wavefront, 
            self.Domain)
    
    def ApplyLateralShift(self, lateralShiftX: float, lateralShiftY: float): 
        if not (isinstance(lateralShiftX, int) or isinstance(lateralShiftX, float)): 
            raise TypeError('The lateralShiftX must be an int or float.')
        if not (isinstance(lateralShiftY, int) or isinstance(lateralShiftY, float)): 
            raise TypeError('The lateralShiftY must be an int or float.')
        
        if self.Domain == Domains.K: 
            wavefront = self.Wavefront + (lateralShiftX * self.X + lateralShiftY * self.Y)
            return MeshTransversalField(
                self.Wavelength, 
                self.Dispersion, 
                self.X, self.Y, 
                self.FieldX, self.FieldY, wavefront, 
                self.Domain)
        else: 
            raise ValueError('Not implemented.')
        
    def CalculateCircularPolarisation(self): 
        fieldL = (self.FieldX + 1.0j * self.FieldY) / np.sqrt(2)
        fieldR = (self.FieldX - 1.0j * self.FieldY) / np.sqrt(2)

        return MeshTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, 
            self.Y, 
            fieldL, 
            fieldR, 
            self.Wavefront, 
            self.Domain)
    
    def TransformToRegular(self, windowSize: float, nPixels: int): 
        if not isinstance(windowSize, float): 
            raise TypeError('The window size must be a float.')
        if windowSize <= 0.0: 
            raise ValueError('The window size must be strictly positive.')
        if not isinstance(nPixels, int): 
            raise TypeError('The number of pixels must be an int.')
        if nPixels <= 0: 
            raise ValueError('The number of pixels must be strictly positive.')
        
        if nPixels % 2 == 0: 
            nPixels += 1

        x = np.linspace(-0.5*windowSize, 0.5*windowSize, nPixels)
        y = np.linspace(-0.5*windowSize, 0.5*windowSize, nPixels)
        x, y = np.meshgrid(x, y, indexing='ij')

        fieldX = griddata((self.X, self.Y), self.FieldX, (x, y))
        fieldY = griddata((self.X, self.Y), self.FieldY, (x, y))
        wavefront = griddata((self.X, self.Y), self.Wavefront, (x, y))

        fieldX *= np.exp(1.0j * wavefront)
        fieldY *= np.exp(1.0j * wavefront)

        fieldX[np.isnan(fieldX)] = 0.0
        fieldY[np.isnan(fieldY)] = 0.0

        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            x, y, 
            fieldX, fieldY, 
            self.Domain)
    
    def TransformToRegularGrid(self, x: np.ndarray, y: np.ndarray): 
        if not isinstance(x, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(x.type, float): 
            raise TypeError()
        if not isinstance(y, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(y.dtype, float): 
            raise TypeError()
        if np.shape(y) != np.shape(x): 
            raise ValueError()
        
        fieldX = griddata((self.X, self.Y), self.FieldX, (x, y))
        fieldY = griddata((self.X, self.Y), self.FieldY, (x, y))
        wavefront = griddata((self.X, self.Y), self.Wavefront, (x, y))

        fieldX *= np.exp(1.0j * wavefront)
        fieldY *= np.exp(1.0j * wavefront)

        fieldX[np.isnan(fieldX)] = 0.0
        fieldY[np.isnan(fieldY)] = 0.0
        
        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            x, y, 
            fieldX, fieldY, 
            self.Domain)
        
    
    def CalculateEz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('The Ez component can only be calculated if the field is defined in the K domain.')
        
        ez = -(self.X * self.FieldX + self.Y * self.FieldY) / self.Kz
        ez[np.isnan(ez)] = 0.0
        
        return MeshElectricField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, self.Y, 
            self.FieldX, self.FieldY, ez, 
            self.Wavefront, 
            self.Domain)

    def DiscardOuterPoints(self, radiusToDiscard: float): 
        if not isinstance(radiusToDiscard, float): 
            raise TypeError()
        if radiusToDiscard <= 0.0: 
            raise ValueError()
        rho = np.sqrt(np.power(self.X,2) + np.power(self.Y,2))
        return MeshTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            self.X[rho<=radiusToDiscard], 
            self.Y[rho<=radiusToDiscard], 
            self.FieldX[rho<=radiusToDiscard], 
            self.FieldY[rho<=radiusToDiscard], 
            self.Wavefront[rho<=radiusToDiscard], 
            self.Domain)
    
    def InterpolateOnGrid(self, windowSize: float, pixels: int): 
        if not isinstance(windowSize, float): 
            raise TypeError()
        if windowSize <= 0.0: 
            raise ValueError()
        if not isinstance(pixels, int): 
            raise TypeError()
        if pixels <= 0: 
            raise ValueError()
        
        if pixels % 2 == 0: 
            pixels += 1
        
        xmin = self.CoordinateOfCentralPixelX - 0.5*windowSize
        xmax = self.CoordinateOfCentralPixelX + 0.5*windowSize
        ymin = self.CoordinateOfCentralPixelY - 0.5*windowSize
        ymax = self.CoordinateOfCentralPixelY + 0.5*windowSize

        aperture = np.nanmax([np.abs(xmin), np.abs(xmax), np.abs(ymin), np.abs(ymax)])

        x = np.linspace(-aperture, aperture, pixels)
        y = np.linspace(-aperture, aperture, pixels)
        x, y = np.meshgrid(x, y, indexing='ij')

        fieldX = griddata((self.X, self.Y), self.FieldX, (x, y))
        fieldY = griddata((self.X, self.Y), self.FieldY, (x, y))
        wavefront = griddata((self.X, self.Y), self.Wavefront, (x, y))

        fieldX[np.isnan(fieldX)] = 0.0
        fieldY[np.isnan(fieldY)] = 0.0
        wavefront[np.isnan(wavefront)] = 0.0
        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            x, y, 
            fieldX, fieldY, 
            self.Domain), wavefront
    
class Sampling:
    @staticmethod
    def ConstructMesh(radius: float, meshLevels: int):
        if not isinstance(radius, float):
            raise TypeError('The radius of the coordinate support should be a float.')
        if radius <= 0.0:
            raise ValueError('The radius of the support should be positive.')
        if not isinstance(meshLevels, int):
            raise TypeError('The number of radial levels in the mesh should be an integer.')
        if meshLevels <= 0:
            raise ValueError('The number of radial levels in the mesh should be positive.')
        
        xmesh = [0.0]
        ymesh = [0.0]

        dr = radius / meshLevels

        for iLevel in range (1, meshLevels):
            for i in range (0, 6*iLevel):
                xmesh.append(iLevel * dr * np.cos(2.0 * np.pi * i / (6.0 * iLevel)))
                ymesh.append(iLevel * dr * np.sin(2.0 * np.pi * i / (6.0 * iLevel)))
        
        xmesh = np.array(xmesh)
        ymesh = np.array(ymesh)

        return xmesh, ymesh

class RegularElectricField(FieldContainer): 
    def __init__(
            self, 
            wavelength: float, 
            dispersion: Material, 
            x: np.ndarray, 
            y: np.ndarray, 
            fieldX: np.ndarray, 
            fieldY: np.ndarray, 
            fieldZ: np.ndarray, 
            domain: Domains
    ): 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength should be a float.')
        if wavelength <= 0.0: 
            raise ValueError('The wavelength should be strictly positive.')
        if not isinstance(dispersion, Material): 
            raise TypeError()
        if not dispersion.IsWavelengthWithinRange(wavelength): 
            raise ValueError()
        if not isinstance(x, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(x.dtype, float): 
            raise TypeError()
        if not isinstance(y, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(y.dtype, float): 
            raise TypeError()
        if np.shape(y) != np.shape(x): 
            raise ValueError()
        if not isinstance(fieldX, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(fieldX.dtype, complex): 
            raise TypeError()
        if np.shape(fieldX) != np.shape(x): 
            raise ValueError()
        if not isinstance(fieldY, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(fieldY.dtype, complex): 
            raise TypeError()
        if np.shape(fieldY) != np.shape(x): 
            raise ValueError()
        if not isinstance(fieldZ, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(fieldZ.dtype, complex): 
            raise TypeError()
        if np.shape(fieldZ) != np.shape(x): 
            raise ValueError()
        if not isinstance(domain, Domains): 
            raise TypeError()
        self._wavelength = wavelength
        self._dispersion = dispersion
        self._x = x
        self._y = y
        self._fieldX = fieldX
        self._fieldY = fieldY
        self._fieldZ = fieldZ
        self._domain = domain
    
    @property
    def Dispersion(self): 
        return self._dispersion
    @Dispersion.setter
    def Dispersion(self, value): 
        if not isinstance(value, Material): 
            raise TypeError()
        self._dispersion = value

    @property
    def Wavelength(self): 
        return self._wavelength
    @Wavelength.setter
    def Wavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The wavelength must be a float.')
        if value <= 0.0: 
            raise ValueError('The wavelength must be strictly positive.')
        if not self.Dispersion.IsWavelengthWithinRange(value): 
            raise ValueError('The wavelength must be within the allowed range by the dispersion characteristics of the medium in which the field is defined.')
        self._wavelength = value
    
    @property
    def X(self): 
        return self._x
    @X.setter
    def X(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._x = value

    @property
    def Y(self): 
        return self._y
    @Y.setter
    def Y(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._y = value

    @property
    def FieldX(self): 
        return self._fieldX
    @FieldX.setter
    def FieldX(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldX = value
    
    @property
    def FieldY(self): 
        return self._fieldY
    @FieldY.setter
    def FieldY(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldY = value

    @property
    def FieldZ(self): 
        return self._fieldZ
    @FieldZ.setter
    def FieldZ(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldZ = value
    
    @property
    def Domain(self): 
        return self._domain
    @Domain.setter
    def Domain(self, value): 
        if not isinstance(value, Domains): 
            raise TypeError()
        self._domain = value
    
    @property
    def Extent(self): 
        return np.array([np.nanmin(self.X), np.nanmax(self.X), np.nanmin(self.Y), np.nanmax(self.Y)])
    
    @property
    def WindowX(self): 
        return np.nanmax(self.X) - np.nanmin(self.X)
    
    @property
    def WindowY(self): 
        return np.nanmax(self.Y) - np.nanmin(self.Y)
    
    @property
    def PixelsX(self): 
        return np.shape(self.X)[0]
    
    @property
    def PixelsY(self): 
        return np.shape(self.X)[1]
    
    @property
    def IndexOfCentralPixelX(self): 
        return int(self.PixelsX/2)
    
    @property
    def IndexOfCentralPixelY(self): 
        return int(self.PixelsY/2)
    
    @property
    def CoordinateOfCentralPixelX(self): 
        return self.X[self.IndexOfCentralPixelX, self.IndexOfCentralPixelY]
    
    @property
    def CoordinateOfCentralPixelY(self): 
        return self.Y[self.IndexOfCentralPixelX, self.IndexOfCentralPixelY]
    
    @property
    def XMin(self): 
        return np.nanmin(self.X)
    
    @property
    def XMax(self): 
        return np.nanmax(self.X)
    
    @property
    def YMin(self): 
        return np.nanmin(self.Y)
    
    @property
    def YMax(self): 
        return np.nanmax(self.Y)
    
    @property
    def PitchX(self): 
        return self.X[1,0] - self.X[0,0]
    
    @property
    def PitchY(self): 
        return self.Y[0,1] - self.Y[0,0]
    
    @property
    def K0(self): 
        return 2.0 * np.pi / self.Wavelength
    
    @property
    def AngularFrequency(self): 
        return self.K0 * Constants.C
    
    @property
    def Wavenumber(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetValue(self.Wavelength)
    
    @property
    def WavenumberReal(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength)
    
    @property
    def Kz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('Kz can only be calculated when the field is defined in the K domain.')
        kz = np.zeros_like(self.X)
        kz = np.sqrt(
            np.power(self.Wavenumber, 2) - 
            np.power(self.X, 2) - 
            np.power(self.Y, 2), dtype=complex)
        kz[np.isnan(kz)] = 0.0
        return kz
    
    @property
    def RealKz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('Kz can only be calculated when the field is defined in the K domain.')
        return np.real(self.Kz)
    
    @property
    def ParaxialIntensity(self): 
        return np.power(np.abs(self.FieldX), 2) + np.power(np.abs(self.FieldY), 2)
    
    @property
    def Intensity(self): 
        return np.power(np.abs(self.FieldX), 2) + np.power(np.abs(self.FieldY), 2) + np.power(np.abs(self.FieldZ), 2)
    
    def CentredExtraction(self, windowFactor): 
        if not (isinstance(windowFactor, float) or isinstance(windowFactor, int)): 
            raise TypeError('The window factor for the extraction must be delivered as an int or float.')
        if windowFactor <= 0.0: 
            raise ValueError('The window factor must be strictly positive.')
        if windowFactor > 1.0: 
            raise ValueError('With a window factor larger than 1 the operation you want is not an extraction. You want to embed the field instead.')
        
        # Calculate number of pixels in zoomed field
        nx = int(self.PixelsX * windowFactor)
        ny = int(self.PixelsY * windowFactor)

        # Ensure that number of pixels is odd
        if nx%2 == 0: 
            nx += 1
        if ny%2 == 0: 
            ny += 1
        
        # Create containers for zoomed field
        xZoom = np.zeros((nx, ny), dtype=float)
        yZoom = np.zeros((nx, ny), dtype=float)
        fieldXZoom = np.zeros((nx, ny), dtype=complex)
        fieldYZoom = np.zeros((nx, ny), dtype=complex)
        fieldZZoom = np.zeros((nx, ny), dtype=complex)

        if windowFactor == 1.0: 
            return self
        else: 
            # Calculate indices of range
            startPixelX = self.IndexOfCentralPixelX - int(nx/2)
            endPixelX = self.IndexOfCentralPixelX + int(nx/2) + 1
            startPixelY = self.IndexOfCentralPixelY - int(ny/2)
            endPixelY = self.IndexOfCentralPixelY + int(ny/2) + 1
            xZoom = self.X[startPixelX:endPixelX, startPixelY:endPixelY]
            yZoom = self.Y[startPixelX:endPixelX, startPixelY:endPixelY]
            fieldXZoom = self.FieldX[startPixelX:endPixelX, startPixelY:endPixelY]
            fieldYZoom = self.FieldY[startPixelX:endPixelX, startPixelY:endPixelY]
            fieldZZoom = self.FieldZ[startPixelX:endPixelX, startPixelY:endPixelY]
        
        return RegularElectricField(
            self.Wavelength, 
            self.Dispersion, 
            xZoom, yZoom, 
            fieldXZoom, fieldYZoom, fieldZZoom,
            self.Domain)
    
    def CentredEmbedding(self, windowFactor): 
        if not (isinstance(windowFactor, int) or isinstance(windowFactor, float)): 
            raise TypeError('The window factor for the embedding must be delivered as an int or float.')
        if windowFactor < 1.0: 
            raise ValueError('For a window factor between 0 and 1, use extraction instead.')
        
        # Calculate number of pixels in embedded field
        windowx = windowFactor * self.WindowX
        windowy = windowFactor * self.WindowY

        nx = int(round(windowx / self.PitchX) + 1)
        ny = int(round(windowy / self.PitchY) + 1)

        # Ensure that number of pixels is odd
        if nx%2 == 0: 
            nx += 1
        if ny%2 == 0: 
            ny += 1
        
        # Recalculate window size 
        windowx = (nx-1) * self.PitchX
        windowy = (ny-1) * self.PitchY
        
        # Create containers for embedded field
        xEmbed = np.zeros((nx, ny), dtype=float)
        yEmbed = np.zeros((nx, ny), dtype=float)
        fieldXEmbed = np.zeros((nx, ny), dtype=complex)
        fieldYEmbed = np.zeros((nx, ny), dtype=complex)
        fieldZEmbed = np.zeros((nx, ny), dtype=complex)

        # Create coordinates
        xEmbed = np.linspace(-0.5*windowx, 0.5*windowx, nx)
        yEmbed = np.linspace(-0.5*windowy, 0.5*windowy, ny)
        xEmbed, yEmbed = np.meshgrid(xEmbed, yEmbed, indexing='ij')

        # Embed field
        startPixelX = int(nx/2) - int(self.PixelsX/2)
        endPixelX = int(nx/2) + int(self.PixelsX/2) + 1
        startPixelY = int(ny/2) - int(self.PixelsY/2)
        endPixelY = int(ny/2) + int(self.PixelsY/2) + 1

        fieldXEmbed[startPixelX:endPixelX, startPixelY:endPixelY] = self.FieldX
        fieldYEmbed[startPixelX:endPixelX, startPixelY:endPixelY] = self.FieldY
        fieldZEmbed[startPixelX:endPixelX, startPixelY:endPixelY] = self.FieldZ

        return RegularElectricField(
            self.Wavelength, 
            self.Dispersion, 
            xEmbed, yEmbed, 
            fieldXEmbed, fieldYEmbed, fieldZEmbed,
            self.Domain)
    
    def Display(self, **optionalArguments): 
        allowedOptionalArguments = {
            'figureSize', 
            'colourMapAmplitude', 
            'colourMapPhase', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks',
            'windowSizeFactor', 
            'normaliseColourScale'
        }
        unrecognisedKeys = set(optionalArguments) - allowedOptionalArguments
        if unrecognisedKeys:
            raise TypeError(f'Unexpected keyword arguments: {", ".join(unrecognisedKeys)}')

        # Retrieve optional arguments
        figureSize = optionalArguments.get('figureSize', Defaults.defaultFieldFigureSize)
        colourMapAmplitude = optionalArguments.get('colourMapAmplitude', Defaults.defaultAmplitudeColourMap)
        colourMapPhase = optionalArguments.get('colourMapPhase', Defaults.defaultPhaseColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        windowSizeFactor = optionalArguments.get('windowSizeFactor', 1.0)
        normaliseColourScale = optionalArguments.get('normaliseColourScale', True)

        normAmplitude = colours.Normalize(
            vmin=0.0, 
            vmax=np.nanmax((np.abs(self.FieldX), np.abs(self.FieldY), np.abs(self.FieldZ))))
        normPhase = colours.Normalize(vmin=-np.pi, vmax=np.pi)

        if windowSizeFactor == 1.0: 
            field = self
        elif windowSizeFactor > 1.0: 
            field = self.CentredEmbedding(windowSizeFactor)
        elif windowSizeFactor < 1.0: 
            field = self.CentredExtraction(windowSizeFactor)
        else: 
            raise ValueError('No other option possible')
        
        if self.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif self.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(self.X, self.Y)
        else: 
            raise ValueError('No other possibility.')

        fig, axs = plt.subplots(
            nrows=2, ncols=3, 
            figsize=figureSize,
            sharex=True, sharey=True)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        if normaliseColourScale:
            cax11 = axs[0][0].imshow(
                np.abs(field.FieldX).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                norm=normAmplitude, 
                origin='lower')
            cax12 = axs[0][1].imshow(
                np.abs(field.FieldY).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                norm=normAmplitude, 
                origin='lower')
            cax13 = axs[0][2].imshow(
                np.abs(field.FieldZ).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                norm=normAmplitude, 
                origin='lower')
        else: 
            cax11 = axs[0][0].imshow(
                np.abs(field.FieldX).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                origin='lower')
            cax12 = axs[0][1].imshow(
                np.abs(field.FieldY).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                origin='lower')
            cax13 = axs[0][2].imshow(
                np.abs(field.FieldZ).T, 
                extent=factorExtent*field.Extent, 
                cmap=colourMapAmplitude, 
                origin='lower')
        cax21 = axs[1][0].imshow(
            np.angle(field.FieldX).T, 
            extent=factorExtent*field.Extent, 
            cmap=colourMapPhase, 
            norm=normPhase, 
            origin='lower')
        cax22 = axs[1][1].imshow(
            np.angle(field.FieldY).T, 
            extent=factorExtent*field.Extent, 
            cmap=colourMapPhase, 
            norm=normPhase, 
            origin='lower')
        cax23 = axs[1][2].imshow(
            np.angle(field.FieldZ).T, 
            extent=factorExtent*field.Extent, 
            cmap=colourMapPhase, 
            norm=normPhase, 
            origin='lower')
        cbar11 = fig.colorbar(cax11)
        cbar12 = fig.colorbar(cax12)
        cbar13 = fig.colorbar(cax13)
        cbar21 = fig.colorbar(cax21)
        cbar22 = fig.colorbar(cax22)
        cbar23 = fig.colorbar(cax23)
        axs[0][0].set_title('$|E_x|$', fontname=font, fontsize=fontsizeTitle)
        axs[0][1].set_title('$|E_y|$', fontname=font, fontsize=fontsizeTitle)
        axs[0][2].set_title('$|E_z|$', fontname=font, fontsize=fontsizeTitle)
        axs[1][0].set_title('arg($E_x$) [rad]', fontname=font, fontsize=fontsizeTitle)
        axs[1][1].set_title('arg($E_y$) [rad]', fontname=font, fontsize=fontsizeTitle)
        axs[1][2].set_title('arg($E_z$) [rad]', fontname=font, fontsize=fontsizeTitle)
        for ax in axs.flat: 
            ax.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
            ax.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        plt.tight_layout()
        del field
        return fig

    def DisplayParaxialIntensity(self, **optionalArguments): 
        allowedParameters = {
            'figureSize', 
            'colourMapIntensity', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks', 
            'windowSizeFactor'
        }
        unrecognisedArguments = set(optionalArguments) - allowedParameters
        if unrecognisedArguments:
            raise TypeError(f'Not recognised: {", ".join(unrecognisedArguments)}.')
        
        # Retrieve optional parameters
        figureSize = optionalArguments.get('figureSize', Defaults.defaultSquareFigureSize)
        colourMapIntensity = optionalArguments.get('colourMapIntensity', Defaults.defaultIntensityColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        windowSizeFactor = optionalArguments.get('windowSizeFactor', 1.0)

        if windowSizeFactor == 1.0: 
            field = self
        elif windowSizeFactor < 1.0: 
            field = self.CentredExtraction(windowSizeFactor)
        elif windowSizeFactor > 1.0: 
            field = self.CentredEmbedding(windowSizeFactor)
        else: 
            raise ValueError('Not possible!')

        if field.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif field.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(field.X, field.Y)
        
        normIntensity = colours.Normalize(vmin=0.0, vmax=np.nanmax(field.ParaxialIntensity))

        fig, main_ax = plt.subplots(figsize=figureSize)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        divider = make_axes_locatable(main_ax)
        top_ax = divider.append_axes('top', 2.0, pad=0.3, sharex=main_ax)
        right_ax = divider.append_axes('right', 2.0, pad=0.3, sharey=main_ax)

        top_ax.xaxis.set_tick_params(labelbottom=False)
        right_ax.yaxis.set_tick_params(labelleft=False)

        main_ax.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        main_ax.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        top_ax.set_ylabel('$\\sum |E_\\perp|^2$', fontname=font, fontsize=fontsizeAxes)
        right_ax.set_xlabel('$\\sum |E_\\perp|^2$', fontname=font, fontsize=fontsizeAxes)
        main_ax.imshow(field.ParaxialIntensity.T, cmap=colourMapIntensity, extent=factorExtent*np.array(field.Extent), norm=normIntensity, origin='lower')
        main_ax.autoscale(enable=False)
        right_ax.autoscale(enable=False)
        top_ax.autoscale(enable=False)
        right_ax.set_xlim(right=np.nanmax(field.ParaxialIntensity))
        top_ax.set_ylim(top=np.nanmax(field.ParaxialIntensity))
        v_line = main_ax.axvline(factorExtent*field.CoordinateOfCentralPixelX, color='white', linestyle='-.', linewidth=1)
        h_line = main_ax.axhline(factorExtent*field.CoordinateOfCentralPixelY, color='white', linestyle='-.', linewidth=1)
        v_prof = right_ax.plot(field.ParaxialIntensity[int(field.PixelsX/2),:], factorExtent*field.Y[int(field.PixelsX/2),:], color='black', linewidth=2)
        h_prof = top_ax.plot(factorExtent*field.X[:, int(field.PixelsY/2)], field.ParaxialIntensity[:, int(field.PixelsY/2)], color='black', linewidth=2)
        del field
        return fig
    
    def DisplayIntensity(self, **optionalArguments): 
        allowedParameters = {
            'figureSize', 
            'colourMapIntensity', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks', 
            'windowSizeFactor'
        }
        unrecognisedArguments = set(optionalArguments) - allowedParameters
        if unrecognisedArguments:
            raise TypeError(f'Not recognised: {", ".join(unrecognisedArguments)}.')
        
        # Retrieve optional parameters
        figureSize = optionalArguments.get('figureSize', Defaults.defaultSquareFigureSize)
        colourMapIntensity = optionalArguments.get('colourMapIntensity', Defaults.defaultIntensityColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        windowSizeFactor = optionalArguments.get('windowSizeFactor', 1.0)

        if windowSizeFactor == 1.0: 
            field = self
        elif windowSizeFactor < 1.0: 
            field = self.CentredExtraction(windowSizeFactor)
        elif windowSizeFactor > 1.0: 
            field = self.CentredEmbedding(windowSizeFactor)
        else: 
            raise ValueError('Not possible!')

        if field.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif field.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(field.X, field.Y)
        
        normIntensity = colours.Normalize(vmin=0.0, vmax=np.nanmax(field.Intensity))

        fig, main_ax = plt.subplots(figsize=figureSize)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        divider = make_axes_locatable(main_ax)
        top_ax = divider.append_axes('top', 2.0, pad=0.3, sharex=main_ax)
        right_ax = divider.append_axes('right', 2.0, pad=0.3, sharey=main_ax)

        top_ax.xaxis.set_tick_params(labelbottom=False)
        right_ax.yaxis.set_tick_params(labelleft=False)

        main_ax.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        main_ax.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        top_ax.set_ylabel('$\\sum |E_\\perp|^2$', fontname=font, fontsize=fontsizeAxes)
        right_ax.set_xlabel('$\\sum |E_\\perp|^2$', fontname=font, fontsize=fontsizeAxes)
        main_ax.imshow(field.Intensity.T, cmap=colourMapIntensity, extent=factorExtent*np.array(field.Extent), norm=normIntensity, origin='lower')
        main_ax.autoscale(enable=False)
        right_ax.autoscale(enable=False)
        top_ax.autoscale(enable=False)
        right_ax.set_xlim(right=np.nanmax(field.Intensity))
        top_ax.set_ylim(top=np.nanmax(field.Intensity))
        v_line = main_ax.axvline(factorExtent*field.CoordinateOfCentralPixelX, color='white', linestyle='-.', linewidth=1)
        h_line = main_ax.axhline(factorExtent*field.CoordinateOfCentralPixelY, color='white', linestyle='-.', linewidth=1)
        v_prof = right_ax.plot(field.Intensity[int(field.PixelsX/2),:], factorExtent*field.Y[int(field.PixelsX/2),:], color='black', linewidth=2)
        h_prof = top_ax.plot(factorExtent*field.X[:, int(field.PixelsY/2)], field.Intensity[:, int(field.PixelsY/2)], color='black', linewidth=2)
        del field
        return fig
    
    def TransformToMesh(self, xmesh: np.ndarray, ymesh: np.ndarray): 
        if not isinstance(xmesh, np.ndarray): 
            raise TypeError('The x coordinates of the point cloud must be delivered as a numpy array.')
        if not isinstance(ymesh, np.ndarray): 
            raise TypeError('The y coordinates of the point cloud must be delivered as a numpy array.')
        if not np.issubdtype(xmesh.dtype, float): 
            raise TypeError('The dtype of the x coordinates of the point cloud must be float.')
        if not np.issubdtype(ymesh.dtype, float): 
            raise TypeError('The dtype of the y coordinates of the point cloud must be float.')
        
        # Check overlap between supports
        if np.nanmax(xmesh) <= self.XMin or np.nanmin(xmesh) >= self.XMax or np.nanmax(ymesh) <= self.YMin or np.nanmin(ymesh) >= self.YMax: 
            raise ValueError('There is no overlap between the supports.')
        
        # fieldX = griddata((self.X.ravel(), self.Y.ravel()), self.FieldX.ravel(), (xmesh, ymesh))
        # fieldY = griddata((self.X.ravel(), self.Y.ravel()), self.FieldY.ravel(), (xmesh, ymesh))

        x = self.X[:,0]
        y = self.Y[0,:]

        interpolatorFieldX = RegularGridInterpolator((x, y), self.FieldX, method='linear', bounds_error=False, fill_value=0.0)
        interpolatorFieldY = RegularGridInterpolator((x, y), self.FieldY, method='linear', bounds_error=False, fill_value=0.0)
        interpolatorFieldZ = RegularGridInterpolator((x, y), self.FieldZ, method='linear', bounds_error=False, fill_value=0.0)

        fieldX = interpolatorFieldX(np.column_stack((xmesh, ymesh)))
        fieldY = interpolatorFieldY(np.column_stack((xmesh, ymesh)))
        fieldZ = interpolatorFieldZ(np.column_stack((xmesh, ymesh)))

        wavefront = np.zeros_like(xmesh)

        return MeshElectricField(
            self.Wavelength, 
            self.Dispersion, 
            xmesh, ymesh, 
            fieldX, fieldY, fieldZ,  
            wavefront, 
            self.Domain)
    
    def Propagate(self, propagationDistance: float): 
        if not (isinstance(propagationDistance, int) or isinstance(propagationDistance, float)): 
            raise TypeError('The propagation distance must be an int or float.')
        if self.Domain != Domains.K: 
            raise ValueError('The field must be defined in the K domain before being propagated.')
        fieldX = self.FieldX * np.exp(1.0j * self.Kz * propagationDistance)
        fieldY = self.FieldY * np.exp(1.0j * self.Kz * propagationDistance)
        fieldZ = self.FieldZ * np.exp(1.0j * self.Kz * propagationDistance)
        return RegularTransversalField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, self.Y, 
            fieldX, fieldY, fieldZ, 
            self.Domain)
    
    def ApplyLateralShift(self, lateralShiftX: float, lateralShiftY: float): 
        if not isinstance(lateralShiftX, float): 
            raise TypeError('The lateralShiftX must be a float.')
        if not isinstance(lateralShiftY, float): 
            raise TypeError('The lateralShiftY must be a float.')
        
        if self.Domain == Domains.K: 
            fieldX = self.FieldX * np.exp(1.0j * (lateralShiftX * self.X + lateralShiftY * self.Y))
            fieldY = self.FieldY * np.exp(1.0j * (lateralShiftX * self.X + lateralShiftY * self.Y))
            fieldZ = self.FieldZ * np.exp(1.0j * (lateralShiftX * self.X + lateralShiftY * self.Y))
            return RegularTransversalField(
                self.Wavelength, 
                self.Dispersion, 
                self.X, self.Y, 
                fieldX, fieldY, fieldZ, 
                self.Domain)
        else: 
            raise ValueError('Only K domain implemented so far.')

class MeshElectricField(FieldContainer): 
    def __init__(
            self, 
            wavelength: float,
            dispersion: Material, 
            x: np.ndarray, 
            y: np.ndarray, 
            fieldX: np.ndarray, 
            fieldY: np.ndarray, 
            fieldZ: np.ndarray, 
            wavefront: np.ndarray, 
            domain: Domains
    ): 
        if not isinstance(wavelength, float): 
            raise TypeError('The wavelength must be a float.')
        if wavelength <= 0.0: 
            raise ValueError('The wavelength must be strictly positive.')
        if not isinstance(dispersion, Material): 
            raise TypeError()
        if not dispersion.IsWavelengthWithinRange(wavelength): 
            raise ValueError('The wavelength must be within the allowed range.')
        if not isinstance(x, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(x.dtype, float): 
            raise TypeError()
        if not isinstance(y, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(y.dtype, float): 
            raise TypeError()
        if np.shape(y) != np.shape(x): 
            raise ValueError()
        if not isinstance(fieldX, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(fieldX.dtype, complex): 
            raise TypeError()
        if np.shape(fieldX) != np.shape(x): 
            raise ValueError()
        if not isinstance(fieldY, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(fieldY.dtype, complex): 
            raise TypeError()
        if np.shape(fieldY) != np.shape(x): 
            raise ValueError()
        if not isinstance(fieldZ, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(fieldZ.dtype, complex):
            raise TypeError()
        if np.shape(fieldZ) != np.shape(x): 
            raise ValueError()
        if not isinstance(wavefront, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(wavefront.dtype, float): 
            raise TypeError()
        if np.shape(wavefront) != np.shape(x): 
            raise ValueError()
        if not isinstance(domain, Domains): 
            raise TypeError()
        self._wavelength = wavelength
        self._dispersion = dispersion
        self._x = x
        self._y = y
        self._fieldX = fieldX
        self._fieldY = fieldY
        self._fieldZ = fieldZ
        self._wavefront = wavefront
        self._domain = domain

    @property
    def Dispersion(self): 
        return self._dispersion
    @Dispersion.setter
    def Dispersion(self, value): 
        if not isinstance(value, Material): 
            raise TypeError('The dispersion properties of the material in which the field is defined must be delivered as an instance of a class derived from Material.')
        self._dispersion = value
    
    @property
    def Wavelength(self): 
        return self._wavelength
    @Wavelength.setter
    def Wavelength(self, value): 
        if not isinstance(value, float): 
            raise TypeError('The wavelength must be a float.')
        if value <= 0.0: 
            raise ValueError('The wavelength must be strictly positive.')
        if not self.Dispersion.IsWavelengthWithinRange(value): 
            raise ValueError('The wavelength must be within the allowed range.')
        self._wavelength = value
    
    @property
    def X(self): 
        return self._x
    @X.setter
    def X(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._x = value

    @property
    def Y(self): 
        return self._y
    @Y.setter
    def Y(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._y = value

    @property
    def FieldX(self): 
        return self._fieldX
    @FieldX.setter
    def FieldX(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldX = value
    
    @property
    def FieldY(self): 
        return self._fieldY
    @FieldY.setter
    def FieldY(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldY = value

    @property
    def FieldZ(self): 
        return self._fieldZ
    @FieldZ.setter
    def FieldZ(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, complex): 
            raise TypeError()
        self._fieldZ = value
    
    @property
    def Wavefront(self): 
        return self._wavefront
    @Wavefront.setter
    def Wavefront(self, value): 
        if not isinstance(value, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(value.dtype, float): 
            raise TypeError()
        self._wavefront = value
    
    @property
    def Domain(self): 
        return self._domain
    @Domain.setter
    def Domain(self, value): 
        if not isinstance(value, Domains): 
            raise TypeError()
        self._domain = value
    
    @property
    def Extent(self): 
        return np.array([np.nanmin(self.X), np.nanmax(self.X), np.nanmin(self.Y), np.nanmax(self.Y)])
    
    @property
    def WindowX(self): 
        return np.nanmax(self.X) - np.nanmin(self.X)
    
    @property
    def WindowY(self): 
        return np.nanmax(self.Y) - np.nanmin(self.Y)
    
    @property
    def Pixels(self): 
        return len(self.X)
    
    @property
    def CoordinateOfCentralPixelX(self): 
        return self.X[0]
    
    @property
    def CoordinateOfCentralPixelY(self): 
        return self.Y[0]
    
    @property
    def XMin(self): 
        return np.nanmin(self.X)
    
    @property
    def XMax(self): 
        return np.nanmax(self.X)
    
    @property
    def YMin(self): 
        return np.nanmin(self.Y)
    
    @property
    def YMax(self): 
        return np.nanmax(self.Y)
    
    @property
    def K0(self): 
        return 2.0 * np.pi / self.Wavelength
    
    @property
    def AngularFrequency(self): 
        return self.K0 * Constants.C
    
    @property
    def Wavenumber(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetValue(self.Wavelength)
    
    @property
    def WavenumberReal(self): 
        return self.K0 * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength)
    
    @property
    def Kz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('Kz can only be calculated when the field is defined in the K domain.')
        kz = np.zeros_like(self.X)
        kz = np.sqrt(
            np.power(self.Wavenumber, 2) - 
            np.power(self.X, 2) - 
            np.power(self.Y, 2), dtype=complex)
        kz[np.isnan(kz)] = 0.0
        return kz
    
    @property
    def RealKz(self): 
        if self.Domain != Domains.K: 
            raise ValueError('Kz can only be calculated when the field is defined in the K domain.')
        return np.real(self.Kz)
    
    @property
    def ParaxialIntensity(self): 
        return np.power(np.abs(self.FieldX), 2) + np.power(np.abs(self.FieldY), 2)
    
    @property
    def WavefrontRange(self): 
        return np.nanmax(self.Wavefront) - np.nanmin(self.Wavefront)
    
    def Display(self, **optionalArguments): 
        allowedArguments = {
            'figureSize', 
            'colourMapAmplitude', 
            'colourMapPhase', 
            'colourMapWavefront', 
            'title', 
            'font', 
            'fontsizeSuptitle', 
            'fontsizeTitle', 
            'fontsizeAxes', 
            'fontsizeTicks', 
            'normaliseColourScale'
        }
        unrecognisedArguments = set(optionalArguments) - allowedArguments
        if unrecognisedArguments: 
            raise TypeError(f'Unrecognised arguments: {", ".join(unrecognisedArguments)}.')
        
        figureSize = optionalArguments.get('figureSize', Defaults.defaultFieldFigureSize)
        colourMapAmplitude = optionalArguments.get('colourMapAmplitude', Defaults.defaultAmplitudeColourMap)
        colourMapPhase = optionalArguments.get('colourMapPhase', Defaults.defaultPhaseColourMap)
        colourMapWavefront = optionalArguments.get('colourMapWavefront', Defaults.defaultWavefrontColourMap)
        title = optionalArguments.get('title', '')
        font = optionalArguments.get('font', Defaults.defaultFont)
        fontsizeSuptitle = optionalArguments.get('fontsizeSuptitle', Defaults.defaultFontsizeSuptitle)
        fontsizeTitle = optionalArguments.get('fontsizeTitle', Defaults.defaultFontsizeTitle)
        fontsizeAxes = optionalArguments.get('fontsizeAxes', Defaults.defaultFontsizeAxes)
        fontsizeTicks = optionalArguments.get('fontsizeTicks', Defaults.defaultAxisTicks)
        normaliseColourScale = optionalArguments.get('normaliseColourScale', True)

        normAmplitude = colours.Normalize(
            vmin=0.0, 
            vmax=np.nanmax((np.abs(self.FieldX), np.abs(self.FieldY), np.abs(self.FieldZ))))
        normPhase = colours.Normalize(vmin=-np.pi, vmax=np.pi)

        if self.Domain == Domains.K: 
            labelX = '$k_x$ [1/m]'
            labelY = '$k_y$ [1/m]'
            factorExtent = 1.0
        elif self.Domain == Domains.X: 
            labelX, labelY, factorExtent = Miscellanea.GetLabelAndExtentFactor(self.X, self.Y)
        
        if self.Domain == Domains.K: 
            propagationCircle11 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle12 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle13 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle21 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle22 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircle23 = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)
            propagationCircleWa = patches.Circle(
                [0.0, 0.0], 
                radius = 2.0 * np.pi * self.Dispersion.RefractiveIndex.GetRealValue(self.Wavelength) / self.Wavelength, 
                edgecolor='magenta', 
                facecolor='none', 
                linewidth=1)

        fig = plt.figure(figsize=figureSize)
        fig.suptitle(title, fontname=font, fontsize=fontsizeSuptitle)
        gs = GridSpec(nrows=2, ncols=5, figure=fig)
        ax11 = plt.subplot(gs.new_subplotspec((0,0)))
        ax12 = plt.subplot(gs.new_subplotspec((0,1)))
        ax13 = plt.subplot(gs.new_subplotspec((0,2)))
        ax21 = plt.subplot(gs.new_subplotspec((1,0)))
        ax22 = plt.subplot(gs.new_subplotspec((1,1)))
        ax23 = plt.subplot(gs.new_subplotspec((1,2)))
        axwa = plt.subplot(gs.new_subplotspec((1,3), colspan=2, rowspan=2))
        if normaliseColourScale:
            cax11 = ax11.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldX), cmap=colourMapAmplitude, norm=normAmplitude, s=10)
            cax12 = ax12.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldY), cmap=colourMapAmplitude, norm=normAmplitude, s=10)
        else: 
            cax11 = ax11.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldX), cmap=colourMapAmplitude, s=10)
            cax12 = ax12.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldY), cmap=colourMapAmplitude, s=10)
        cax13 = ax13.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.abs(self.FieldZ), cmap=colourMapAmplitude, norm=normAmplitude, s=10)
        cax21 = ax21.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.angle(self.FieldX), cmap=colourMapPhase, norm=normPhase, s=10)
        cax22 = ax22.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.angle(self.FieldY), cmap=colourMapPhase, norm=normPhase, s=10)
        cax23 = ax23.scatter(factorExtent*self.X, factorExtent*self.Y, c=np.angle(self.FieldZ), cmap=colourMapPhase, norm=normPhase, s=10)
        caxwa = axwa.scatter(factorExtent*self.X, factorExtent*self.Y, c=self.Wavefront, cmap=colourMapWavefront, s=10)
        if self.Domain == Domains.K and (self.WindowX > 0.5*self.Wavenumber or self.WindowY > 0.5*self.Wavenumber):
            cax11Circle = ax11.add_patch(propagationCircle11)
            cax12Circle = ax12.add_patch(propagationCircle12)
            cax13Circle = ax13.add_patch(propagationCircle13)
            cax21Circle = ax21.add_patch(propagationCircle21)
            cax22Circle = ax22.add_patch(propagationCircle22)
            cax23Circle = ax23.add_patch(propagationCircle23)
            caxWaCircle = axwa.add_patch(propagationCircleWa)
        ax11.axis('equal')
        ax12.axis('equal')
        ax13.axis('equal')
        ax21.axis('equal')
        ax22.axis('equal')
        ax23.axis('equal')
        axwa.axis('equal')
        ax11.set_title('$|E_x|$', fontname=font, fontsize=fontsizeTitle)
        ax12.set_title('$|E_y|$', fontname=font, fontsize=fontsizeTitle)
        ax13.set_title('$|E_z|$', fontname=font, fontsize=fontsizeTitle)
        ax21.set_title('arg($E_x$) [rad]', fontname=font, fontsize=fontsizeTitle)
        ax22.set_title('arg($E_y$) [rad]', fontname=font, fontsize=fontsizeTitle)
        ax23.set_title('arg($E_z$) [rad]', fontname=font, fontsize=fontsizeTitle)
        axwa.set_title('Wavefront [rad]', fontname=font, fontsize=fontsizeTitle)
        axwa.text(np.nanmin(factorExtent*self.X), np.nanmin(factorExtent*self.Y), f'{self.WavefrontRange:.2f} rad', fontname=font, fontsize=fontsizeAxes)
        cbar11 = fig.colorbar(cax11)
        cbar12 = fig.colorbar(cax12)
        cbar13 = fig.colorbar(cax13)
        cbar21 = fig.colorbar(cax21)
        cbar22 = fig.colorbar(cax22)
        cbar23 = fig.colorbar(cax23)
        cbarwa = fig.colorbar(caxwa)
        ax11.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        ax21.set_ylabel(labelY, fontname=font, fontsize=fontsizeAxes)
        ax21.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        ax22.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        ax23.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        axwa.set_xlabel(labelX, fontname=font, fontsize=fontsizeAxes)
        plt.tight_layout()
        return fig

    def Propagate(self, propagationDistance): 
        if not (isinstance(propagationDistance, int) or isinstance(propagationDistance, float)): 
            raise TypeError('The propagation distance must be delivered as an int or float.')
        if self.Domain != Domains.K: 
            raise ValueError('The field can only be propagated if it is defined in the K domain.')
        
        wavefront = self.Wavefront + self.Kz * propagationDistance
        return MeshElectricField(
            self.Wavelength, 
            self.Dispersion, 
            self.X, self.Y, 
            self.FieldX, self.FieldY, self.FieldZ, 
            wavefront, 
            self.Domain)  

    def ApplyLateralShift(self, lateralShiftX: float, lateralShiftY: float): 
        if not (isinstance(lateralShiftX, int) or isinstance(lateralShiftX, float)): 
            raise TypeError('The lateralShiftX must be an int or float.')
        if not (isinstance(lateralShiftY, int) or isinstance(lateralShiftY, float)): 
            raise TypeError('The lateralShiftY must be an int or float.')
        
        if self.Domain == Domains.K: 
            wavefront = self.Wavefront + (lateralShiftX * self.X + lateralShiftY * self.Y)
            return MeshElectricField(
                self.Wavelength, 
                self.Dispersion, 
                self.X, self.Y, 
                self.FieldX, self.FieldY, self.FieldZ, 
                wavefront, 
                self.Domain)
        else: 
            raise ValueError('Not implemented.')  
        
    def TransformToRegular(self, windowSize: float, nPixels: int): 
        if not isinstance(windowSize, float): 
            raise TypeError('The window size must be a float.')
        if windowSize <= 0.0: 
            raise ValueError('The window size must be strictly positive.')
        if not isinstance(nPixels, int): 
            raise TypeError('The number of pixels must be an int.')
        if nPixels <= 0: 
            raise ValueError('The number of pixels must be strictly positive.')
        
        if nPixels % 2 == 0: 
            nPixels += 1

        x = np.linspace(-0.5*windowSize, 0.5*windowSize, nPixels)
        y = np.linspace(-0.5*windowSize, 0.5*windowSize, nPixels)
        x, y = np.meshgrid(x, y, indexing='ij')

        fieldX = griddata((self.X, self.Y), self.FieldX, (x, y))
        fieldY = griddata((self.X, self.Y), self.FieldY, (x, y))
        fieldZ = griddata((self.X, self.Y), self.FieldZ, (x, y))
        wavefront = griddata((self.X, self.Y), self.Wavefront, (x, y))

        fieldX = fieldX * np.exp(1.0j * wavefront)
        fieldY = fieldY * np.exp(1.0j * wavefront)
        fieldZ = fieldZ * np.exp(1.0j * wavefront)

        fieldX[np.isnan(fieldX)] = 0.0
        fieldY[np.isnan(fieldY)] = 0.0
        fieldZ[np.isnan(fieldZ)] = 0.0

        return RegularElectricField(
            self.Wavelength, 
            self.Dispersion, 
            x, y, 
            fieldX, fieldY, fieldZ, 
            self.Domain)
    
    def TransformToRegularGrid(self, x: np.ndarray, y: np.ndarray): 
        if not isinstance(x, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(x.type, float): 
            raise TypeError()
        if not isinstance(y, np.ndarray): 
            raise TypeError()
        if not np.issubdtype(y.dtype, float): 
            raise TypeError()
        if np.shape(y) != np.shape(x): 
            raise ValueError()
        
        fieldX = griddata((self.X, self.Y), self.FieldX, (x, y))
        fieldY = griddata((self.X, self.Y), self.FieldY, (x, y))
        fieldZ = griddata((self.X, self.Y), self.FieldZ, (x, y))
        wavefront = griddata((self.X, self.Y), self.Wavefront, (x, y))

        fieldX *= np.exp(1.0j * wavefront)
        fieldY *= np.exp(1.0j * wavefront)
        fieldZ *= np.exp(1.0j * wavefront)

        fieldX[np.isnan(fieldX)] = 0.0
        fieldY[np.isnan(fieldY)] = 0.0
        fieldZ[np.isnan(fieldZ)] = 0.0
        
        return RegularElectricField(
            self.Wavelength, 
            self.Dispersion, 
            x, y, 
            fieldX, fieldY, fieldZ, 
            self.Domain)

    def DiscardOuterPoints(self, radiusToDiscard: float): 
        if not isinstance(radiusToDiscard, float): 
            raise TypeError()
        if radiusToDiscard <= 0.0: 
            raise ValueError()
        rho = np.sqrt(np.power(self.X,2) + np.power(self.Y,2))
        return MeshElectricField(
            self.Wavelength, 
            self.Dispersion, 
            self.X[rho<=radiusToDiscard], 
            self.Y[rho<=radiusToDiscard], 
            self.FieldX[rho<=radiusToDiscard], 
            self.FieldY[rho<=radiusToDiscard], 
            self.FieldZ[rho<=radiusToDiscard],
            self.Wavefront[rho<=radiusToDiscard], 
            self.Domain)

