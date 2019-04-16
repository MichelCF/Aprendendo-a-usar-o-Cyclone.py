#https://github.com/matplotlib/basemap/issues/419
import cyclone as cn
import xarray as xr
import numpy as np
from metpy import calc


PNMM = xr.open_dataset('PNMM.2010.nc')
vento_U = xr.open_dataset("U_10M.2010.nc")
vento_V = xr.open_dataset("V_10M.2010.nc")
lat = vento_U.latitude
#print((lat.values))
lon = vento_V.longitude
#print((lon.values))
threshold=[1010.,0.0004,0.0003,1300.,3000.]
sw=np.array([275.,275.])

#vorticidade = calc.vorticity(vento_U.u10,vento_V.v10,0.75,0.75)



candidatos = cn.findcandidates(lon.values,lat.values, PNMM.msl, 0,0,threshold,sw)

#print(candidatos)



