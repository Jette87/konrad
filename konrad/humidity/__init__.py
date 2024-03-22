"""This module contains classes for handling humidity."""
import logging
import warnings

from konrad.component import Component
from konrad.utils import prefix_dict_keys
from konrad.physics import relative_humidity2vmr, vmr2relative_humidity, integrate_vmr
from .stratosphere import *
from .relative_humidity import *


logger = logging.getLogger(__name__)


class FixedRH(Component):
    """Preserve the relative humidity profile under temperature changes."""

    def __init__(self, rh_func=None, stratosphere_coupling=None):
        """Create a humidity handler.

        Parameters:
            rh_func (callable): Callable that describes the vertical
                relative humidity distribution.
                If `None`, assume a :class:`VerticallyUniform` relative humidity.
            stratosphere_coupling (callable): Callable that describes how the
                humidity should be treated in the stratosphere.
        """
        if stratosphere_coupling is None:
            self._stratosphere_coupling = ColdPointCoupling()
        else:
            self._stratosphere_coupling = stratosphere_coupling

        if rh_func is None:
            self._rh_func = VerticallyUniform()
        else:
            self._rh_func = rh_func

        self._rh_profile = None

    @property
    def netcdf_subgroups(self):
        return {
            "rh_func": self._rh_func,
            "stratosphere_coupling": self._stratosphere_coupling,
        }

    def hash_attributes(self):
        # Make sure that non-``Component`` attributes do not break hashing.
        return hash(
            tuple(
                attr.hash_attributes()
                for attr in (self._rh_func, self._stratosphere_coupling)
                if hasattr(attr, "hash_attributes")
            )
        )

    @property
    def rh_func(self):
        return type(self._rh_func).__name__

    @property
    def stratosphere_coupling(self):
        return type(self._stratosphere_coupling).__name__

    def adjust_humidity(self, atmosphere, **kwargs):
        """Determine the humidity profile based on atmospheric state.

        Parameters:
            TODO: Write docstring.

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        atmosphere["H2O"][-1, :] = relative_humidity2vmr(
            relative_humidity=self._rh_func(atmosphere, **kwargs),
            pressure=atmosphere["plev"],
            temperature=atmosphere["T"][-1],
        )
        self._stratosphere_coupling.adjust_stratospheric_vmr(atmosphere)

class FixedIWV(FixedRH):
    """ keep the integrated water vapor constant"""
    def __init__(self,rh_func=None,  stratosphere_coupling=None, IWV=None, **kwargs):
        super().__init__(rh_func=rh_func, stratosphere_coupling=stratosphere_coupling)
        self.IWV=IWV
        
    
    def rescale_IWV(self, atmosphere, **kwargs):
        IWV_curr = integrate_vmr(vmr=atmosphere["H2O"][-1, :], pressure=atmosphere["plev"])
        print(IWV_curr, self.IWV)
        if not self.IWV:
            self.IWV = IWV_curr
            warnings.warn("no changed IWV. Same as fixed RH calculation")
        #norm atmospheric water vapor by the ratio of integrated water vapor
        atmosphere["H2O"][-1, :] = atmosphere["H2O"][-1, :] * self.IWV / IWV_curr
        
class ChangeRHwithT(FixedIWV):
    def __init__(self, rh_func=None, stratosphere_coupling=None,f_T=None, **kwargs):
        if not f_T:
            f_T = lambda x: 0
            
        def new_rh(atmosphere, T_diff):
            return rh_func(atmosphere) + f_T(atmosphere)*T_diff
            
        super().__init__(rh_func=lambda atm, T_diff: new_rh(atm, T_diff), stratosphere_coupling=stratosphere_coupling, **kwargs)

        
    def adjust_humidity(self, atmosphere, T_diff, **kwargs):
        rh = self._rh_func(atmosphere,T_diff, **kwargs) #+ self._f_T(atmosphere['T'][-1]) * T_diff
        
        atmosphere['H2O'][-1, :] = relative_humidity2vmr(rh, 
                                                         pressure = atmosphere['plev'],
                                                         temperature = atmosphere['T'][-1])
        self._stratosphere_coupling.adjust_stratospheric_vmr(atmosphere)
        
        
class ChangeRHwithp(FixedIWV):
    def __init__(self, rh_func=None, stratosphere_coupling=None,f_p=None, **kwargs):
        super().__init__(rh_func=rh_func, stratosphere_coupling=stratosphere_coupling, **kwargs)
        if f_T:
            self._f_p = f_p
        else:
            self._f_p = lambda x: 0
        
    def adjust_humidity(self, atmosphere, T_diff, **kwargs):
        rh = self._rh_func(atmosphere, **kwargs) + self._f_p(atmosphere['plev'][-1]) * T_diff
        
        atmosphere['H2O'][-1, :] = relative_humidity2vmr(rh, 
                                                         pressure = atmosphere['plev'],
                                                         temperature = atmosphere['T'][-1])
        self._stratosphere_coupling.adjust_stratospheric_vmr(atmosphere)
        
                
        
        
class FixedVMR(Component):
    """Keep the water vapor volume mixing ratio constant."""

    def __init__(self, *args, **kwargs):
        if len(args) + len(kwargs) > 0:
            # Allow arguments to be passed for consistent interface but
            # warn the user.
            logger.warning(f"All input arguments to {self} are ignored.")

        # Set both attributes for consistent user interface and netCDF output.
        self.rh_func = "FixedVMR"
        self.stratosphere_coupling = "FixedVMR"

    def adjust_humidity(self, atmosphere, **kwargs):
        return
