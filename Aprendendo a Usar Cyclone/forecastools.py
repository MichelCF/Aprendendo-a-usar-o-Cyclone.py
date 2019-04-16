#!/usr/bin/python
#
# ForecasTools
#
# Group of functions to handle and work with NOAA forecast products
#
# See the few lines bellow to check python dependencies. An easy way to install it is by using python anaconda:
# https://store.continuum.io/cshop/anaconda/
# which can be installed without administration permission. After instalation and after adding anaconda path to .bashrc or equivalent:
# conda install numpy
# conda install basemap
# conda install pylab
# conda install matplotlib
# conda install netcdf4
# conda install ftplib
# 
# Functions defined below:
# forecastools.getnoaaforecast    # to fetch NOAA forecast fields using nomads OpenDAP
# forecastools.getnoaaforecastp   # to fetch NOAA forecast on a single point using nomads OpenDAP
# forecastools.plotforecast       # to plot fields or grid point time-series
# forecastools.uploadit           # to upload figures to server/website via ftp
# forecastools.list               # list of the most important url and useful variables from NOAA nomads
# forecastools.distance           # calculate distance in km between two points with lat1,lon1 and lat2,lon2
#
#
#   Copyright (C) 2017 ATMOSMARINE TECNOLOGIA E CONSULTORIA LTDA
#
#   Contact:
#   AtmosMarine Development Branch, atm@atmosmarine.com
#
#   Provided under license for
#   ATMOSMARINE TECNOLOGIA E CONSULTORIA LTDA, atm@atmosmarine.com
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   A copy of the GNU General Public License is provided at
#   http://www.gnu.org/licenses/
#
#	version 1.1:    13/10/2015
#	version 1.2:    08/04/2016
#	version 1.3:    02/08/2017 (minor corrections to run using python version 3)
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Contributions: Anaconda Community (https://www.continuum.io/anaconda-community), MotorDePopa Wave Research Group (motordepopa@googlegroups.com) 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from matplotlib import ticker
import netCDF4
from time import gmtime, strftime, time, mktime
import sys
import pylab
from pylab import *
import copy
import datetime
from datetime import *
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
colormap = cm.GMT_polar
palette = plt.cm.jet
palette.set_bad('aqua', 10.0)
import ftplib
from matplotlib.mlab import *



def getnoaaforecast(*args):
	'''
	Function to fetch NOAA forecast using nomads OpenDAP
	[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)

	4 inputs are requested:
	nomads : vector string with the fixed http://nomads address string, model type (one or two) and variable name
		For Ex: nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.glo_30mext','htsgwsfc']
		        nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','nww3','htsgwsfc']
			nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
	initime : vector of initial time with year, month, day and hour (optional)
		For Ex: initime=[2015,10,13]
			initime=[2015,10,13,6]
			aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
	nt : number of forecast timesteps to download, maximum of 81
		For Ex: nt=9 # one day forecast (GFS 3-hourly)
			nt=1 # only the first step
	lonlat : vector with initial and final longitude and latitude
		For Ex: lonlat=[-180,180,-90,90]
			lonlat=[0,360,-90,90]
			lonlat=[20,120,-80,20]
	levels (optional) : atmospheric level in case parameter has multi-levels. It does not need to be informed if variable has only one level, you can omit it.
		 levels are (in mb): [ 1000.,   975.,   950.,   925.,   900.,   850.,   800.,   750., 700.,   
					650.,   600.,   550.,   500.,   450.,   400.,   350.,   300.,   250.,   
					200.,   150.,   100.,    70.,    50.,    30.,  20.,    10.]
		 For Ex: levels=[1000,850,700]

	4 outputs are provided;
	data : matrix of data related to the variable name and nomads address informed
		data.shape = [nt,lat.shape,lon.shape]  or  data.shape = [nt,levels,lat.shape,lon.shape]
	lon : longitude vector
	lat : latitude vector
	ntime : time vector:  gmtime, seconds since 1979/01/01 . Converted from original NOAA nomads ("days since 1-1-1 00:00:0.0") using ntime[0]*3600.*24 -(719437*3600*24)
		with gmtime(ntime[0]) you can confirm the exact date

	check http://nomads.ncep.noaa.gov:9090/ for information about specific paths, files and variables names
	Atmospheric results from GFS 0.25 at
	http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/    where ugrd10m and vgrd10m are the U and V wind components at 10 meters
	Wave Multigrid results from WAVEWATCH III at
	http://nomads.ncep.noaa.gov:9090/dods/wave/mww3/   where htsgwsfc is the total significant wave height, dirpwsfc is the peak direction and perpwsfc is the peak period
	Pay attention to the rtofs directory products (circulation model), it takes more time to make the forecast available than the other models!

	forecastools.list()  gives also some useful links and mostly used variables names

	Examples:

		import forecastools 
		from time import gmtime, strftime, time 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.glo_30mext','htsgwsfc'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=9 
		lonlat=[-180,180,-90,90] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 

		import forecastools 
		from time import gmtime, strftime, time 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=57 
		lonlat=[20,120,-80,20] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 

		import forecastools 
		from time import gmtime, strftime, time 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','nww3','htsgwsfc'] 
		initime=[2016,4,8]   # set a new date here or you will get an error message 
		nt=4 # forecast at 12Z 
		lonlat=[-80,-10, -80, 10] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 

		import forecastools 
		from time import gmtime, strftime, time 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods', 'gfs_0p50', 'ugrdprs']  # 'ugrdprs' is a multi-level parameter 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10]),6] 
		nt=1 
		lonlat=[-180, 180, -90, 90] 
		levels=[1000,850,700] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels) # data has one more dimension of levels 

		import forecastools 
		from time import gmtime, strftime, time 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods', 'rtofs', 'sst']  # sea surface temperature, Real-Time Ocean Forecast System 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1 
		lonlat=[-90, 20, -82, 10] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		# Pay attention to the rtofs directory products (circulation model), it takes more time to make the forecast available than the other models! 

	version 1.1:    13/10/2015
	version 1.2:    08/04/2016
        www.atmosmarine.com
	'''

	if len(args) < 4:
		sys.exit(' ERROR! Insuficient input arguments. 4 variable inputs must be entered: nomads, initime, nt, lonlat, \
			 You can enter 0 or '' in case you do not know the variables initime, nt or lonlat ')

	if len(args) == 4:
		nomads=copy.copy(args[0]); initime=copy.copy(args[1]); nt=copy.copy(args[2]); lonlat=copy.copy(args[3]);
	elif len(args) == 5:
		nomads=copy.copy(args[0]); initime=copy.copy(args[1]); nt=copy.copy(args[2]); lonlat=copy.copy(args[3]); levels=copy.copy(args[4]);

	if '' in nomads:	
		nomads.remove('')

	if nomads==0 or nomads=='':
		sys.exit(' ERROR! Insuficient input arguments in nomads. nomads=[fixdomain,modeltype,modelgrid,variable] or nomads=[fixdomain,modeltype,variable]')
	elif len(nomads)<3:
		sys.exit(' ERROR! Insuficient input arguments in nomads. nomads=[fixdomain,modeltype,modelgrid,variable] or nomads=[fixdomain,modeltype,variable]')

	if np.atleast_1d(initime).shape[0]<3:
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		print([' Warning. Creating initial date, initime = '+repr(initime)])
	if nt==0 or nt=='':
		nt=81
		print([' Warning. Creating number of timesteps, nt = '+repr(nt)])


	# gridrf : grid referential for large domains, if longitudes are from -180 to 180, gridrf=1. If longitudes are from 0 to 360, gridrf=2.
	if np.atleast_1d(lonlat).shape[0] < 4:
		lonlat = [-180,180,-90,90]; gridrf = 1
		print([' Warning. Creating global initial and final longitudes and latitudes as, lonlat = '+repr(lonlat)])
		print(' In case a small sub-domain model is downloaded, these initial and final longitudes and latitudes will be modified') 
	elif np.atleast_1d(lonlat).shape[0] == 4:
		if lonlat[0]<0. and lonlat[1]<181. :
			gridrf = 1
		elif lonlat[0]>=0. and lonlat[1]<361.:
			gridrf = 2
		else:	
			sys.exit(' ERROR! Wrong values of longitudes in lonlat. It must be within the interval between [-180 to 180] or [0 to 360].')
	else:
		sys.exit(' ERROR! Wrong lonlat length. It must provide four values of initial and final longitudes and latitudes')


	if lonlat[2]<-90 or lonlat[3]>90 :
		sys.exit(' ERROR! Wrong lonlat arguments. Latitudes must be between -90 and 90')

	# Initial date setup
	initime=np.array(initime)
	if initime.shape[0]==3:
		initime=np.append(initime,0)
	elif initime[3] not in [0,6,12,18]:
		sys.exit(' ERROR! Hour must be 00, 06, 12 or 18')

	# date string and hour based on initime
	strdate=str(initime[0]).zfill(4)+str(initime[1]).zfill(2)+str(initime[2]).zfill(2)
	strhour=str(initime[3]).zfill(2)

	# a simple solution in case input nomads address ends with / or not.
	if nomads[0][-1] == '/':
		nomads[0] = nomads[0][0:-1]

	if nomads[1]=='mww3' and len(nomads)<4:
		sys.exit(' ERROR! Insuficient input arguments in nomads in multigrid (mww3), nomads=[fixdomain,modeltype,modelgrid,variable]')

	# organize url when nomads full address has 4 or 3 parameters: fixed http://nomads address string, model type (one or two) and variable name
	if len(nomads) > 3: 
		url=nomads[0]+'/'+nomads[1]+'/'+strdate+'/'+nomads[2]+strdate+'_'+strhour+'z'
	else:
		if nomads[1][0:3]=='gfs':
			auxs=nomads[1][0:3]
			url=nomads[0]+'/'+nomads[1]+'/'+auxs+strdate+'/'+nomads[1]+'_'+strhour+'z'
		elif nomads[1]=='ice':
			url=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+strdate+'/'+nomads[1]+'.'+strhour+'z'
		elif nomads[1][0:5]=='rtofs':
			nomads[1]=nomads[1][0:5]
			url=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+'_global'+strdate+'/'+nomads[1]+'_glo_2ds_forecast_3hrly_prog'
			urlt0=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+'_global'+strdate+'/'+nomads[1]+'_glo_2ds_nowcast_3hrly_prog'
		else:
			url=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+strdate+'/'+nomads[1]+strdate+'_'+strhour+'z'


	try:
		# open netcdf file, Check if url exist 
		nfile = netCDF4.Dataset(url)

	except:
		sys.exit([' ERROR when fetching data: '+url+' Does not exist or file still not available'])
	else:
		# url exists, moving on...

		if nomads[1]=='rtofs':
			nfilet0 = netCDF4.Dataset(urlt0)

		# time and latitude
		ntime = nfile.variables['time'][0:nt]
		nlat  = nfile.variables['lat'][:]

		# Indexes of initial and final latitudes selected
		indlati=find(abs(nlat-lonlat[2])==min(abs(nlat-lonlat[2]))).min()
		indlatf=find(abs(nlat-lonlat[3])==min(abs(nlat-lonlat[3]))).max()
		nlat = nlat[indlati:indlatf+1]

		# In case of many Z-levels
		if len(args) == 5:
			levlist=nfile.variables['lev'][:]
			indlevels=[]		
			for i in range(0,len(np.atleast_1d(levels))): 
				if np.atleast_1d(levels)[i] in levlist:
					indlevels=np.append(indlevels,int(find(np.array(levlist[:])==float(np.atleast_1d(levels)[i]))[0]))
				else:
					sys.exit('Input level requested '+repr(np.atleast_1d(levels)[i])+' does not exist in the file')

		# wait until data is obtained 
		data = [] ; cd = 1
		while len(data) == 0 and cd<100 :

			# Selecting longitude interval using two referentials, (-180 to 180) or (0 to 360). 
			#Depending on the region it is better to use gridrf=1 or gridrf=2.


			# Real-Time Ocean Forecast System with a different referential
			if nomads[1]=='rtofs':

				alon  = nfile.variables['lon'][:]

				if gridrf==1:

					if lonlat[0]>(alon[np.isnan(alon)==False].min()-360) and lonlat[1]<(alon[np.isnan(alon)==False].max()-360):
						lonlat[0]=lonlat[0]+360; lonlat[1]=lonlat[1]+360
						nlon  = np.copy(alon)
						indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
						indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
						data = nfile.variables[nomads[-1]][0:nt,0,indlati:indlatf+1,indloni:indlonf+1]
						data[0,:,:] = nfilet0.variables[nomads[-1]][-1,0,indlati:indlatf+1,indloni:indlonf+1]
						nlon=nfile.variables['lon'][indloni:indlonf+1]-360
					else:
						data = nfile.variables[nomads[-1]][0:nt,0,indlati:indlatf+1,:]
						data[0,:,:] = nfilet0.variables[nomads[-1]][-1,0,indlati:indlatf+1,:]
						for i in range(0,nt):
							alon=nfile.variables['lon'][:]	
							data[i,:,:],nlon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),data[i,:,:],alon,start=False)

						indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
						indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
						nlon = nlon[indloni:indlonf+1]
						data = data[:,:,indloni:indlonf+1]

				elif gridrf==2:

					if lonlat[0]>alon[np.isnan(alon)==False].min() and lonlat[1]<alon[np.isnan(alon)==False].max():
						nlon  = np.copy(alon)
						indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
						indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
						data = nfile.variables[nomads[-1]][0:nt,0,indlati:indlatf+1,indloni:indlonf+1]
						data[0,:,:] = nfilet0.variables[nomads[-1]][-1,0,indlati:indlatf+1,indloni:indlonf+1]
						nlon=nfile.variables['lon'][indloni:indlonf+1]
					else:
						data = nfile.variables[nomads[-1]][0:nt,0,indlati:indlatf+1,:]
						data[0,:,:] = nfilet0.variables[nomads[-1]][-1,0,indlati:indlatf+1,:]
						for i in range(0,nt):
							alon=nfile.variables['lon'][:]	
							data[i,:,:],nlon = shiftgrid(360.05,data[i,:,:],alon,start=False)

						indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
						indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
						nlon = nlon[indloni:indlonf+1]
						data = data[:,:,indloni:indlonf+1]


			else: 
				# All the other atmospheric and wave products, from 0to360 referential
				if gridrf==1:
					alon  = nfile.variables['lon'][:]
					if (alon[np.isnan(alon)==False].max()-alon[np.isnan(alon)==False].min())>350:
		
						# longitudes from -180 to 180
						nlon  = nfile.variables['lon'][:]-180
						indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
						indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
						nlon = nlon[indloni:indlonf+1]

						if len(nfile.variables[nomads[-1]].shape)==3:			
							data = nfile.variables[nomads[-1]][0:nt,indlati:indlatf+1,:]
							for i in range(0,nt):
								alon=nfile.variables['lon'][:]	
								data[i,:,:],alon = shiftgrid(180.+(np.diff(alon[np.isnan(alon)==False]).mean()/2.),data[i,:,:],alon,start=False)

							data = data[:,:,indloni:indlonf+1]
						elif len(nfile.variables[nomads[-1]].shape)==4:
							data = nfile.variables[nomads[-1]][0:nt,indlevels.astype(int),indlati:indlatf+1,:]
							level=nfile.variables['lev'][:]
							for i in range(0,nt):
								for j in range(0,indlevels.shape[0]):
									alon=nfile.variables['lon'][:]	
									data[i,j,:,:],alon = shiftgrid(180.+(np.diff(alon[np.isnan(alon)==False]).mean()/2.),data[i,j,:,:],alon,start=False)

							data = data[:,:,:,indloni:indlonf+1]

					else:
						# NOAA nomads subgrids (regional grids) longitudes are from 0 to 360, but user can prefer from -180 to 180 
						if lonlat[0]<0:
							lonlat[0]=lonlat[0]+360
							if lonlat[1] <=360: 
								lonlat[1]=lonlat[1]+360
							nlon  = nfile.variables['lon'][:]
							indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
							indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()

							if len(nfile.variables[nomads[-1]].shape)==3:
								data = nfile.variables[nomads[-1]][0:nt,indlati:indlatf+1,indloni:indlonf+1]
							elif len(nfile.variables[nomads[-1]].shape)==4:
								data = nfile.variables[nomads[-1]][0:nt,indlevels.astype(int),indlati:indlatf+1,indloni:indlonf+1]

							nlon=nfile.variables['lon'][indloni:indlonf+1]-360

						else:
							nlon  = nfile.variables['lon'][:]
							indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
							indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
							nlon = nlon[indloni:indlonf+1]

							if len(nfile.variables[nomads[-1]].shape)==3:
								data = nfile.variables[nomads[-1]][0:nt,indlati:indlatf+1,indloni:indlonf+1]
							elif len(nfile.variables[nomads[-1]].shape)==4:
								data = nfile.variables[nomads[-1]][0:nt,indlevels.astype(int),indlati:indlatf+1,indloni:indlonf+1]

				elif gridrf==2:
					# longitudes from 0 to 360
					if lonlat[1]>lonlat[0]:
						nlon  = nfile.variables['lon'][:]
						indloni=find(abs(nlon-lonlat[0])==min(abs(nlon-lonlat[0]))).min()
						indlonf=find(abs(nlon-lonlat[1])==min(abs(nlon-lonlat[1]))).max()
						nlon = nlon[indloni:indlonf+1]

						if len(nfile.variables[nomads[-1]].shape)==3:
							data = nfile.variables[nomads[-1]][0:nt,indlati:indlatf+1,indloni:indlonf+1]
						elif len(nfile.variables[nomads[-1]].shape)==4:
							data = nfile.variables[nomads[-1]][0:nt,indlevels.astype(int),indlati:indlatf+1,indloni:indlonf+1]


			cd=cd+1 

		# Check if download data was properly done or not
		if len(data)>0:
			aux=data[data>-9999.].max()-data[data>-9999.].min()
			if aux>0:
				if len(nfile.variables[nomads[-1]].shape)==4 and nomads[1] != 'rtofs':
					level=nfile.variables['lev'][:]
					print('Data exctracted at specific level(s): '+repr(level[indlevels.astype(int)]))
					print('One more dimension expected in output data allocated after time dimension')

				del nomads, initime, nt, lonlat, gridrf, strdate, strhour, url, nfile, indlati, indlatf, indloni, indlonf
				return data,nlon,nlat,(ntime*3600.*24 -(719164*3600.*24.))
				del data, nlon, nlat, ntime
			else:
				sys.exit(' ERROR fetching file and variable. Lat lon ranges can be out of ranges or you selected a masked area')
		else:
			sys.exit(' ERROR fetching file and variable. Lat lon ranges can be out of ranges or you selected a masked area')


		nfile.close()


def getnoaaforecastp(*args):
	'''
 	Function to fetch single grid point NOAA forecast using nomads OpenDAP
	This is a simple variation of forecastools.getnoaaforecast , restrited to single point request
	[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)

	4 inputs are requested:
	nomads : vector string with the fixed http://nomads address string, model type (one or two) and variable name
		For Ex: nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.glo_30mext','htsgwsfc']
		        nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','nww3','htsgwsfc']
			nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
	initime : vector of initial time with year, month, day and hour (optional)
		For Ex: initime=[2015,10,13]
			initime=[2015,10,13,6]
			aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
	nt : number of future timesteps to download, maximum of 81
		For Ex: nt=9 # one day forecast (GFS 3-hourly)
			nt=1 # only the first step
	lonlat : vector of only one longitude and one latitude
		For Ex: lonlat=[-70,26]
	levels (optional) : atmospheric level in case parameter has multi-levels. It does not need to be informed if variable has only one level, you can omit it.
		 levels are (in mb): [ 1000.,   975.,   950.,   925.,   900.,   850.,   800.,   750., 700.,   
					650.,   600.,   550.,   500.,   450.,   400.,   350.,   300.,   250.,   
					200.,   150.,   100.,    70.,    50.,    30.,  20.,    10.]
		 For Ex: levels=[1000,850,700]

	4 outputs are provided;
	data : matrix of data related to the variable name and nomads address informed
		data.shape = [nt,] or data.shape = [nt,levels] 
	lon : longitude of the exact grid point extracted
	lat : latitude of the exact grid point extracted
	ntime : time vector:  gmtime, seconds since 1979/01/01 . Converted from original NOAA nomads ("days since 1-1-1 00:00:0.0") using ntime[0]*3600.*24 -(719437*3600.*24.)
		with gmtime(ntime[0]) you can confirm the exact date

	ALWAYS CHECK IF YOUR INPUT LONLAT IS THE SAME (OR VERY CLOSE) TO THE OUTPUT LON NLAT !

	see http://nomads.ncep.noaa.gov:9090/ for information about specific paths, files and variables names
	Atmospheric results from GFS 0.25 at
	http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/    where ugrd10m and vgrd10m are the U and V wind components at 10 meters
	Wave Multigrid results from WAVEWATCH III at
	http://nomads.ncep.noaa.gov:9090/dods/wave/mww3/   where htsgwsfc is the total significant wave height, dirpwsfc is the peak direction and perpwsfc is the peak period

	forecastools.list()  gives also some useful links and mostly used variables names

	Examples:

		from time import gmtime, strftime, time 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.at_10m','htsgwsfc'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		lonlat=[-70,26]; nt=81; 
		[data,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 

		from time import gmtime, strftime, time 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=17 # 2 days forecast 
		lonlat=[-49.86667,-31.56667] 
		[uwnd,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m'] 
		[vwnd,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 

		from time import gmtime, strftime, time 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs'] # geopotential height [gpm], multi-level variable 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=41 # 5 days forecast 
		lonlat=[-49.86667,-31.56667] 
		levels=[1000,850,700,500] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat,levels)  # data has one more dimension of levels 

		from time import gmtime, strftime, time 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods', 'rtofs', 'sst']  # sea surface temperature, Real-Time Ocean Forecast System 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=41 # 5 days forecast 
		lonlat=[-49.86667,-31.56667] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 
		# Pay attention to the rtofs directory products (circulation model), it takes more time to make the forecast available than the other models! 

	version 1.1:    15/10/2015
	version 1.2:    08/04/2016
        www.atmosmarine.com
	'''

	if len(args) < 4:
		sys.exit(' ERROR! Insuficient input arguments. 4 inputs must be entered: nomads, initime, nt, lonlat \
			 You can enter 0 or '' in case you do not know variables initime or nt ')

	if len(args) == 4:
		nomads=copy.copy(args[0]); initime=copy.copy(args[1]); nt=copy.copy(args[2]); lonlat=copy.copy(args[3]);
	elif len(args) == 5:
		nomads=copy.copy(args[0]); initime=copy.copy(args[1]); nt=copy.copy(args[2]); lonlat=copy.copy(args[3]); levels=copy.copy(args[4]);

	if '' in nomads:	
		nomads.remove('')

	if nomads==0 or nomads=='':
		sys.exit(' ERROR! Insuficient input arguments in nomads. nomads=[fixdomain,modeltype,modelgrid,variable] or nomads=[fixdomain,modeltype,variable]')
	elif len(nomads)<3:
		sys.exit(' ERROR! Insuficient input arguments in nomads. nomads=[fixdomain,modeltype,modelgrid,variable] or nomads=[fixdomain,modeltype,variable]')

	if np.atleast_1d(initime).shape[0]<3:
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		print([' Warning. Creating initial date, initime = '+repr(initime)])

	# check lonlat vector and gridrf coordinates
	if np.atleast_1d(lonlat).shape[0] != 2:
		sys.exit(' ERROR! Single grid-point only. Wrong lonlat, it must have only one latitude and one longitude. For ex: lonlat=[-40,-25]')
	else:
		if lonlat[0]<0:
			gridrf=1
		else:
			gridrf=2

		lonlat=np.array([lonlat[0]-0.000001,lonlat[0]+0.000001,lonlat[1]-0.000001,lonlat[1]+0.000001])


	# Initial date setup
	initime=np.array(initime)
	if initime.shape[0]==3:
		initime=np.append(initime,0)
	elif initime[3] not in [0,6,12,18]:
		sys.exit(' ERROR! Hour must be 00, 06, 12 or 18')
	# date string and hour based on initime
	strdate=str(initime[0]).zfill(4)+str(initime[1]).zfill(2)+str(initime[2]).zfill(2)
	strhour=str(initime[3]).zfill(2)
	# organize url when nomads full address has 4 or 3 parameters: fixed http://nomads address string, model type (one or two) and variable name
	if len(nomads) > 3: 
		url=nomads[0]+'/'+nomads[1]+'/'+strdate+'/'+nomads[2]+strdate+'_'+strhour+'z'
	else:
		if nomads[1][0:3]=='gfs':
			auxs=nomads[1][0:3]
			url=nomads[0]+'/'+nomads[1]+'/'+auxs+strdate+'/'+nomads[1]+'_'+strhour+'z'
		elif nomads[1]=='ice':
			url=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+strdate+'/'+nomads[1]+'.'+strhour+'z'
		elif nomads[1][0:5]=='rtofs':
			nomads[1]=nomads[1][0:5]
			url=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+'_global'+strdate+'/'+nomads[1]+'_glo_2ds_forecast_3hrly_prog'
		else:
			url=nomads[0]+'/'+nomads[1]+'/'+nomads[1]+strdate+'/'+nomads[1]+strdate+'_'+strhour+'z'


	try:
		# open netcdf file, Check if url exist 
		nfile = netCDF4.Dataset(url)
	except:
		sys.exit([' ERROR when fetching data: '+url+' Does not exist or file still not available'])
	else:
		# url exists, moving on...
		# latitude and longitude ranges
		alon  = nfile.variables['lon'][:]
		alat  = nfile.variables['lat'][:]

		if nomads[1][0:5] != 'rtofs':
			if gridrf==1:
				if lonlat[0]+360 < alon[0] or lonlat[1]+360 > alon[-1]:
					sys.exit([' ERROR! Longitude out of model grid range'])
			else:
				if lonlat[0] < alon[0] or lonlat[1] > alon[-1]:
					sys.exit([' ERROR! Longitude out of model grid range'])

			if lonlat[2] < alat[0] or lonlat[3] > alat[-1]:
				sys.exit([' ERROR! Latitude out of model grid range'])

		nfile.close()
		import forecastools

		# In case of many Z-levels
		if len(args) == 5:
			# using forecastools.getnoaaforecast as explained above
			[data,nlon,nlat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		else:
			[data,nlon,nlat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)

		# error message in case a wrong point is selected.
		if (abs(nlon)-abs(lonlat[0]))<0.51 and (abs(nlat)-abs(lonlat[2]))<0.51:
			if len(data.shape)==3:	
				data=np.copy(data[:,0,0])
			elif len(data.shape)==4:
				data=np.copy(data[:,:,0,0])

			del nomads, nt, lonlat, initime, gridrf, strdate, strhour, url, nfile, alon, alat
			return data,nlon,nlat,ntime;
			del data, nlon, nlat, ntime
		else:
			sys.exit(' ERROR! Wrong user input Lat Lon, out of the domain related to the url model')



def plotforecast(*args):
	'''
	Function to plot fields using python basemap tool - designed to plot NOAA forecast fileds and time-series obtained with function forecastools.getnoaaforecast or forecastools.getnoaaforecastp

	forecastools.plotforecast(data,lat,lon,figinfo)    # 2D matrix (field) with related to a fixed specific time instant (not informed)
	forecastools.plotforecast(data,lat,lon,ntime,figinfo)   # 1D matrix (time-series) with a time vector ntime (need to be informed)

	3 to 5 inputs are requested:
	data : 2-dimensional: matrix with the field intended to plot. It can be also provided as udata, vdata  in case of a vector variable.
		For Ex: swh[:,:]
			uwnd[:,:],vwnd[:,:]
	       1-dimensional: vector (time-series) can be plotted too. In this case latitude must be a single value, as well as longitude, and time array ntime must be informed.
	latitude : latitude vector starting with the further south latitude (pointing north), for the 2D option. Or single value for the 1D option
	longitude : longitude vector starting with the further west longitude, for the 2D option (It can has the referential -180 to 180 or 0 to 360). Or single value for the 1D option
	ntime : time array in case of time-series on specific grid point wanted.  seconds since 1979/01/01. With this option data must be 1-dimensional vector and latitude and longitude must be single values (not vectors)

	Pay attention to the shape of data as well as lat and lon. It switches from 1D to 2D plot and vice-versa depending on the shape of data

	figinfo : vector with information for Basemap plot (in case of 2D option). You do not need to provide it, in this case automatic figure parameters will be calculated
		  It must have 5 arguments ( len(figinfo)=5 ): figure size; levels vector for the colour bar; skip rate (to skip plotting vectors); arrow scale; title
		     levels vector, skip rate and arrow scale can be set as 0 (zero) in case a 1D plot (time-series) is requested 
		  figure size related to x and y dimensions, for example (8,5)
		  levels vector for the colour bar as for example [0,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,1]
		  skip rate must me a single integer, for example 7 
		  arrow scale is also an integer and it controls the arrows lengths
		  title must be an string between '
			skip and arrow scale can be set as 0 in case your input fiels is not a vector (not U and V) but a single variable as Hs.

		  For Ex: [(8,5),[0,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,1],0,0,'NOAA Ice converage']  # for a 2D plot (field)
		      Ex: [(8,5),0,0,0,'NOAA Significant Wave Height (7 days forecast)']  # for a 1D plot (time-series)

	Output consists of a figure that can be saved using:
        	savefig('NameOfInterest.jpg', dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='jpg',transparent=False, bbox_inches=None, pad_inches=0.1)
		see other options of savefig

	- Examples (2D plots/fields):

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; lonlat=[-45,-37,-26,-21]; 
		[uwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m'] 
		[vwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		forecastools.plotforecast(uwnd[0,:,:],vwnd[0,:,:],lat,lon) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','icecsfc'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; lonlat=[-180,180,-80,90]; 
		[ice,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [(8,5),[0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99,1],0,0,'NOAA Ice converage'] 
		forecastools.plotforecast(ice[0,:,:],lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','tmp2m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; lonlat=[-180,180,-90,90]; 
		[t2m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		t2m=np.copy(t2m[0,:,:])-273.15 
		levels = np.linspace( t2m[(np.isnan(t2m)==False) & (t2m>-9999)].min(), np.percentile(t2m[(np.isnan(t2m)==False) & (t2m<9999)],99.9), 20) 
		figinfo = [(8,5),levels,0,0,'NOAA Forecast, Temperature at 2 meters  '+str(gmtime(ntime)[2]).zfill(2)+'/'+str(gmtime(ntime)[1]).zfill(2)+'/'+repr(gmtime(ntime)[0])+' '+str(gmtime(ntime)[3]).zfill(2)+'Z'] 
		forecastools.plotforecast(t2m,lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.glo_30mext','htsgwsfc'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; 
		lonlat=[-180,180,-90,90] # Although multi_1.at_10m is a regional model you can input general large lon-lat domains. In this case the total grid result of multi_1.at_10m domains will be plotted. 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [0,0,0,0,'Significant Wave Height (m)   '+str(gmtime(ntime)[1]).zfill(2)+'/'+str(gmtime(ntime)[2]).zfill(2)+'/'+repr(gmtime(ntime)[0])+' '+str(gmtime(ntime)[3]).zfill(2)+'Z'] # I do not know any figure parameters but I want to add a title. 
		forecastools.plotforecast(data[0,:,:],lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.at_10m','htsgwsfc'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; 
		lonlat=[-180,180,-90,90] # Although multi_1.at_10m is a regional model you can input general large lon-lat domains. In this case the total grid result of multi_1.at_10m domains will be plotted. 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [0,0,0,0,'Significant Wave Height (m)   '+str(gmtime(ntime)[1]).zfill(2)+'/'+str(gmtime(ntime)[2]).zfill(2)+'/'+repr(gmtime(ntime)[0])+' '+str(gmtime(ntime)[3]).zfill(2)+'Z'] # I do not know any figure parameters but I want to add a title. 
		forecastools.plotforecast(data[0,:,:],lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdmwl'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; lonlat=[-35,30,30,80]; 
		[maxuwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdmwl'] 
		[maxvwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [0,0,0,0,'Maximum Wind Speed'] # I do not know any figure parameters but I want to add a title. 
		forecastools.plotforecast(maxuwnd[0,:,:],maxvwnd[0,:,:],lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; lonlat=[75,150,-10,32]; 
		[uwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m'] 
		[vwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [0,0,0,0,'10m Wind Speed'] # I do not know any figure parameters but I want to add a title. 
		forecastools.plotforecast(uwnd[0,:,:],vwnd[0,:,:],lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m'] 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10]),00] 
		nt=9; lonlat=[0,360,-90,90]; 
		[uwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		nomads[2]='vgrd10m' 
		[vwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [(8,5),0,20,800,'Global Wind Speed (m/s) at 10m for tomorrow: '+str(gmtime(ntime[-1])[2]).zfill(2)+'/'+str(gmtime(ntime[-1])[1]).zfill(2)+'/'+repr(gmtime(ntime[-1])[0])+' '+str(gmtime(ntime[-1])[3]).zfill(2)+'Z'] 
		forecastools.plotforecast(uwnd[-1,:,:],vwnd[-1,:,:],lat,lon,figinfo) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods', 'rtofs', 'sst']  # sea surface temperature, Real-Time Ocean Forecast System 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1 
		lonlat=[-90, 20, -82, 10] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [(8,7),0,0,0,'Sea Surface Temperature (C): '+str(gmtime(ntime[-1])[2]).zfill(2)+'/'+str(gmtime(ntime[-1])[1]).zfill(2)+'/'+repr(gmtime(ntime[-1])[0])+' '+str(gmtime(ntime[-1])[3]).zfill(2)+'Z'] 
		forecastools.plotforecast(data[0,:,:],lat,lon,figinfo) 
		# Pay attention to the rtofs directory products (circulation model), it takes more time to make the forecast available than the other models! 


		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1 
		lonlat=[-90, 20, -82, 10] 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods', 'rtofs', 'u_velocity']  # surface current, Real-Time Ocean Forecast System 
		[ucurr,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		nomads[2]='v_velocity'
		[vcurr,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		figinfo = [(8,7),0,0,0,'Surface Current Speed (m/s): '+str(gmtime(ntime[-1])[2]).zfill(2)+'/'+str(gmtime(ntime[-1])[1]).zfill(2)+'/'+repr(gmtime(ntime[-1])[0])+' '+str(gmtime(ntime[-1])[3]).zfill(2)+'Z'] 
		forecastools.plotforecast(ucurr[-1,:,:],vcurr[-1,:,:],lat,lon,figinfo) 
		# Pay attention to the rtofs directory products (circulation model), it takes more time to make the forecast available than the other models! 


	- Examples (1D plots/time-series):

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods/wave','mww3','multi_1.at_10m','htsgwsfc'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		lonlat=[-70,26] ; nt=57; 
		[data,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 
		figinfo = [0,0,0,0,'NOAA Forecast, Significant Wave Height (m)'] 
		forecastools.plotforecast(data,lat,lon,ntime,figinfo) 
		ylabel("Hs (m)", fontsize=8) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10]),00] 
		nt=81; lonlat=[105,6]; 
		[uwnd,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 
		nomads[2]='vgrd10m' 
		[vwnd,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 
		figinfo = [(10,5),0,0,0,'NOAA Forecast, GFS0.25 Wind Speed at 10m at 6N / 105E'] 
		data=np.sqrt(uwnd**2+vwnd**2) 
		forecastools.plotforecast(data,lat,lon,ntime,figinfo) 
		ylabel("Wind Speed (m/s)", fontsize=10) 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs'] # geopotential height [gpm], multi-level variable 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=41; # 5 days forecast 
		lonlat=[-49.86667,-31.56667] 
		levels=[1000,850,700,500] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat,levels) 
		figinfo = [(10,5),0,0,0,'NOAA Forecast, Geopotential Height (850mb) at 31.6S / 49.9W'] 
		forecastools.plotforecast(data[:,1],lat,lon,ntime,figinfo) # data has one more dimension of levels 

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods', 'rtofs', 'sst']  # sea surface temperature, Real-Time Ocean Forecast System 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=61 
		lonlat=[105,6] 
		[data,lon,lat,ntime]=forecastools.getnoaaforecastp(nomads,initime,nt,lonlat) 
		figinfo = [(10,5),0,0,0,'Sea Surface Temperature (C) at 105E / 6N'] 
		forecastools.plotforecast(data[:],lat,lon,ntime,figinfo) 
		# Pay attention to the rtofs directory products (circulation model), it takes more time to make the forecast available than the other models! 


	version 1.1:	15/10/2015
	version 1.2:    08/04/2016
        www.atmosmarine.com
	'''

	if len(args) < 3:
		sys.exit(' ERROR! Insuficient input arguments. At least one matrix (lat X lon), latitude vector and longitude vector must be provided.')

	elif len(args) == 3:
		data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); figinfo=0  # 2D plot with no info about fig parameters
		mtype=1
	elif len(args) == 4: 

		if min(np.atleast_2d(args[0]).shape) > 1: # 2D plot

			if min(np.atleast_2d(args[1]).shape) > 1:  # U and V, fig info not informed
				uwnd=copy.copy(args[0]); vwnd=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=0
				if uwnd.shape != vwnd.shape:
					sys.exit(' ERROR! U and V sizes are different')

				data=np.sqrt((uwnd*uwnd)+(vwnd*vwnd)) # magnitude (in case of two vector components inputs)
				if data.shape[0] != lat.shape[0] or data.shape[1] != lon.shape[0]:
					sys.exit(' ERROR! matrix size is not matching lat and/or lon sizes')

				mtype=2
			else: # data, fig info informed
				data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); figinfo=copy.copy(args[3])
				if data.shape[0] != lat.shape[0] or data.shape[1] != lon.shape[0]:
					sys.exit(' ERROR! matrix size is not matching lat and/or lon sizes')

				mtype=1
				if figinfo != 0:				
					if len(figinfo)<5:
						sys.exit(' ERROR! Insuficient figinfo input arguments. 5 arguments inside figinfo must be provided.')

		else:  # 1D plot, fig info not informed
			data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); ntime=copy.copy(args[3]); figinfo=0

	elif len(args) == 5:

		if min(np.atleast_2d(args[0]).shape) > 1: # 2D plot
			uwnd=copy.copy(args[0]); vwnd=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=copy.copy(args[4])
			if uwnd.shape != vwnd.shape:
				sys.exit(' ERROR! U and V sizes are different')
			data=np.sqrt((uwnd*uwnd)+(vwnd*vwnd)) # magnitude (in case of two vector components inputs).
			if data.shape[0] != lat.shape[0] or data.shape[1] != lon.shape[0]:
				sys.exit(' ERROR! matrix size is not matching lat and/or lon sizes')
			mtype=2
			if figinfo != 0:				
				if len(figinfo)<5:
					sys.exit(' ERROR! Insuficient figinfo input arguments. 5 arguments inside figinfo must be provided.')

		else: # 1D plot
			data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); ntime=copy.copy(args[3]); figinfo=copy.copy(args[4])

	else:
		sys.exit(' ERROR! Too many input arguments. 3 to 5 inputs are necessary only. help(forecastools.plotforecast) for more information.')

	# mask values
	if abs(np.array(data)).max() > 9.99e+20:
		data[abs(data)>9.99e+20]=NaN

	if len(np.atleast_1d(data).shape)>2:
		sys.exit(' ERROR! Wrong input data shape. It must be a one or two-dimensional variable')
	elif len(np.atleast_1d(data).shape)==1:
		# 1D plot, time series plot
		if np.atleast_1d(lat).shape[0] > 1 or np.atleast_1d(lon).shape[0] > 1 or np.atleast_1d(data).shape[0] < 2 :
			sys.exit(' ERROR! Wrong input data shape. Input data is a 1D vector array (time series will be ploted)')
		elif ntime.shape[0] != data.shape[0]:
			sys.exit(' ERROR! Time and data array must have the same length.')
		else:
			if np.atleast_1d(figinfo).shape[0] < 5:
				figinfo = [0,0,0,0,'']
			
			if figinfo[0] != 0:
				fig=plt.figure(figsize=figinfo[0])
			else:
				fig=plt.figure()

			year=np.zeros(ntime.shape[0],'i'); month=np.zeros(ntime.shape[0],'i'); day=np.zeros(ntime.shape[0],'i'); hour=np.zeros(ntime.shape[0],'i'); minute=np.zeros(ntime.shape[0],'i')
			for i in range(0,ntime.shape[0]):
				year[i]=gmtime(ntime[i])[0]; month[i]=gmtime(ntime[i])[1]; day[i]=gmtime(ntime[i])[2]; hour[i]=gmtime(ntime[i])[3]; minute[i]=gmtime(ntime[i])[4];

			dates = np.array([date2num(datetime(yy,mm,dd,hh,mi)) for yy,mm,dd,hh,mi in zip(year,month,day,hour,minute)])
			ax = plt.subplot2grid((7,1), (0,0), rowspan=6); plt.xticks(rotation=45)
			color = '0.75'; ax.plot_date(dates,data,color); plt.grid()
			ax.set_xlim( dates[0], dates[-1] )
			ax.plot_date(dates,data,'ko')
			ax.xaxis.set_major_formatter( DateFormatter('%d-%m-%Y %HZ') )
			plt.ylabel(" ", fontsize=8); plt.xlabel("Time", fontsize=10);plt.title(figinfo[4],fontsize=12)
			for label in ax.get_xticklabels() + ax.get_yticklabels():
				label.set_fontsize(8)

	else:
		# 2D plot, fields plot
		# defining automatic figure parameters in case input is not provided
		if figinfo==0:
			# level for cbar 
			if data[(np.isnan(data)==False)].min() != np.percentile(data[(np.isnan(data)==False)],99.99):
				levels = np.linspace( data[(np.isnan(data)==False)].min(), np.percentile(data[(np.isnan(data)==False)],99.99), 20)
			else:
				levels = np.linspace( data[(np.isnan(data)==False)].min(), data[(np.isnan(data)==False)].max(), 20)
			figinfo = [0,levels,0,0,'']
		#elif figinfo[1]==0:
		elif np.atleast_1d(figinfo[1]).shape[0]==1:
			if data[(np.isnan(data)==False)].min() != np.percentile(data[(np.isnan(data)==False)],99.99):
				levels = np.linspace( data[(np.isnan(data)==False)].min(), np.percentile(data[(np.isnan(data)==False)],99.99), 20)
			else:
				levels = np.linspace( data[(np.isnan(data)==False)].min(), data[(np.isnan(data)==False)].max(), 20)
			figinfo[1] = levels

		# cbar string format
		if (max(figinfo[1])-min(figinfo[1])) > 20:
			nformat='%4.0f'
		elif (max(figinfo[1])-min(figinfo[1])) > 5:
			nformat='%3.1f'
		elif (max(figinfo[1])-min(figinfo[1])) > 0.999:
			nformat='%3.2f'
		elif (max(figinfo[1])-min(figinfo[1])) > 0.1:
			nformat='%3.3f'
		else:
			nformat='%3.4f'

		# opening figure with figure size in case it is informed  
		if figinfo[0] != 0:
			fig=plt.figure(figsize=figinfo[0])
		else:
			fig=plt.figure()

		# Plots (http://matplotlib.org/basemap/users/mapsetup.html)
		# (1) GLOBAL -----------
		if (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())>350:

			if lon[np.isnan(lon)==False].max()>350 :
				if mtype==2:
					uwnd[:,:],alon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),uwnd[:,:],lon,start=False)
					vwnd[:,:],alon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),vwnd[:,:],lon,start=False)
					data[:,:],lon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),data[:,:],lon,start=False)
				else:
					data[:,:],lon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),data[:,:],lon,start=False)

			# Global Plot with Eckert IV Projection	        	
			map = Basemap(projection='eck4',lon_0 = 0, resolution = 'c')
			if np.any(data[np.isnan(data)==True]) or (min(figinfo[1])>data[np.isnan(data)==False].min()-0.001) or (max(figinfo[1])<data[np.isnan(data)==False].max()+0.001):
				map.bluemarble(scale=0.2); map.drawlsmask(ocean_color='w',land_color='None')
			else:
				map.drawlsmask(ocean_color='w',land_color='grey') # map.fillcontinents(color='grey')
			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.contourf(xx,yy,data,figinfo[1],cmap=palette,extend="max")
			map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),40),labels=[0,0,0,1],linewidth=0.3,fontsize=6)
			map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),20),labels=[1,0,0,0],linewidth=0.3,fontsize=6)
			#map.drawstates(linewidth=0.1); #map.fillcontinents(color='gray')
			ax = plt.gca()
			pos = ax.get_position()
			l, b, w, h = pos.bounds
			cax = plt.axes([l+0.07, b+0.01, w-0.15, 0.025]) # setup colorbar axes.
			cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat);# cbar.ax.tick_params(labelsize=fsmpb)
			tick_locator = ticker.MaxNLocator(nbins=8); cbar.locator = tick_locator; cbar.update_ticks()
			plt.axes(ax)  # make the original axes current again
			plt.title(figinfo[4])
			# if it is a vector field, as winds and currents:
			if mtype==2:
				# figure parameter (skip vector) in case it is not provided
				if figinfo[2] ==0: 
					sx = round(13* (float(data[:,:].shape[0])*0.8 / float(data[:,:].shape[1])),1);  sy = 13 - sx
					figinfo[2] = int( (  (7.*(data[:,:].shape[0]+data[:,:].shape[1]) )/ 1080 ) * 41./(sx*sy) ) 
					figinfo[3] = 800 * (data[np.isnan(data)==False].max()/30.)
				[mnlon,mnlat]=np.meshgrid(lon[::figinfo[2]],lat[::figinfo[2]])
				xx, yy = map(mnlon,mnlat)
				Q = map.quiver(xx,yy,uwnd[::figinfo[2],::figinfo[2]],vwnd[::figinfo[2],::figinfo[2]],width=0.001,scale=figinfo[3])

		# (2) Regional covering high latitudes -----------
		elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())>20 and (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())<70 and abs(lat[np.isnan(lat)==False]).max()>60 and (float(lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())/float(lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()))<3. and (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())<60:

			# size/shape of the figure
			nwidth=int( (5000000*(lon[-1]-lon[0]))/50 )
			nheight=int( (5000000*(lat[-1]-lat[0]))/40 )
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )	
			# Stereographic Projection ------------------
			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 50:
				fres='f'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 100:
				fres='h'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 500:
				fres='i'

			map = Basemap(width=nwidth, height=nheight, resolution=fres, projection='stere',
				lat_ts=lat[0], lat_0=((lat[-1]-lat[0])/2)+lat[0], lon_0=((lon[-1]-lon[0])/2)+lon[0]    )

			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100:
				map.bluemarble(scale=np.tanh(4000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')


			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.contourf(xx,yy,data[:,:],figinfo[1],cmap=palette,extend="max")
			# map.drawstates(); map.fillcontinents(color='gray')
			map.drawmeridians(np.arange(round(lon[0]),round(lon[-1]),lonmd),labels=[0,0,0,1],linewidth=0.3)
			map.drawparallels(np.arange(round(lat[0]),round(lat[-1]),latmd),labels=[1,0,0,0],linewidth=0.3)
			ax = plt.gca(); pos = ax.get_position()
			if abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.7:
				db=0.06
			elif abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.5:
				db=0.05
			else:
				db=0.03
			l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
			cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
			tick_locator = ticker.MaxNLocator(nbins=8); cbar.locator = tick_locator; cbar.update_ticks()
			plt.axes(ax)  # make the original axes current again
			plt.title(figinfo[4]) 
			# if it is a vector field, as winds and currents:
			if mtype==2:
				# figure parameter (skip vector) and scale in case it is not provided
				if figinfo[2] == 0:
					npts=(float(data[:,:].shape[0])*(float(data[:,:].shape[1])))
					if npts<300:
						figinfo[2] = 1
					else:
						figinfo[2] = int(pow(npts,0.3)/3)
				figinfo[3]=int(400+pow(npts,0.4))*(data[np.isnan(data)==False].max()/30.)
				# vector grid and plot with quiver 
				[mnlon,mnlat]=np.meshgrid(lon[::figinfo[2]],lat[::figinfo[2]])
				xx, yy = map(mnlon,mnlat)
				Q = map.quiver(xx,yy,uwnd[::figinfo[2],::figinfo[2]],vwnd[::figinfo[2],::figinfo[2]],width=0.003,scale=figinfo[3])

		# (3) Regional -----------
		else:
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )
			# fig parameters
			if lat[0] < 0 and lat[-1] <= 0:
				lat_0=-(abs(lat[-1])+abs(lat[0]))/2.0
			else:
				lat_0=(lat[0]+lat[-1])/2.0
	
			if lon[0] < 0 and lon[-1] <= 0:
				lon_0=-(abs(lon[-1])+abs(lon[0]))/2.0
			else:
				lon_0=(lon[0]+lon[-1])/2.0

			#   Regional, Equidistant Cylindrical Projection ------------------
			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 50:
				fres='f'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 100:
				fres='h'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 500:
				fres='i'

			map = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution=fres)

			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100 and lon[-1]<=180:
				map.bluemarble(scale=np.tanh(5000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')
				#map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)
				#map.fillcontinents(color='grey',lake_color='aqua')

			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)  
			map.contourf(xx,yy,data[:,:],figinfo[1],cmap=palette,extend="max")
			# map.drawstates(); # map.fillcontinents(color='gray')
			map.drawmeridians(np.arange(round(lon.min()),round(lon.max()),lonmd),labels=[0,0,0,1])
			map.drawparallels(np.arange(round(lat.min()),round(lat.max()),latmd),labels=[1,0,0,0])
			ax = plt.gca(); pos = ax.get_position(); 
			if abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.7:
				db=0.06
			elif abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.5:
				db=0.05
			else:
				db=0.03
			l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
			cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
			tick_locator = ticker.MaxNLocator(nbins=8); cbar.locator = tick_locator; cbar.update_ticks()
			plt.axes(ax)  # make the original axes current again
			plt.title(figinfo[4]) 
			# if it is a vector field, as winds and currents:
			if mtype==2:
				# figure parameter (skip vector) and scale in case it is not provided
				npts=(float(data[:,:].shape[0])*(float(data[:,:].shape[1])))
				if figinfo[2] ==0: 
					if npts<300:
						figinfo[2] = 1
					else:
						figinfo[2] = int(pow(npts,0.3)/3)
				figinfo[3]=int(400+pow(npts,0.4))*(data[np.isnan(data)==False].max()/30.)
				# vector grid and plot with quiver
				[mnlon,mnlat]=np.meshgrid(lon[::figinfo[2]],lat[::figinfo[2]])
				xx, yy = map(mnlon,mnlat)
				Q = map.quiver(xx,yy,uwnd[::figinfo[2],::figinfo[2]],vwnd[::figinfo[2],::figinfo[2]],width=0.002,scale=figinfo[3])

	del figinfo, fig





def plotforecasts(*args):
	'''
	This is the same function forecastools.plotforecast but some variables used for the plots are sent out of the function, as:
		map       Basemap plot/projection
		xx, yy,   latitude and longitude point position used for the plots (from meshgrid)
		figinfo,  figure information as described in 
		fwdt      width of vector arrows in map.quiver, to set up plots with 2 components (U and V). It is given only in cases forecastools.plotforecasts had two inputs U and V 
	The plot is produced and shown exactly as forecastools.plotforecast

	If it is a single variable field, results are:
		map, xx, yy, figinfo
	In case two components are entered (U, V) with quiver plots using arrows, results are:
		map, xx, yy, figinfo, xx2, yy2, fwdt
	where xx2 and yy2 are xx and yy but with some skip level and less latitudes and longitudes

	- Examples:

		from time import gmtime, strftime, time 
		from pylab import * 
		import forecastools 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','tmp2m'] 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nt=1; lonlat=[-180,180,-90,90]; 
		[t2m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		t2m=np.copy(t2m[0,:,:])-273.15 
		levels = np.linspace( t2m[(np.isnan(t2m)==False) & (t2m>-9999)].min(), np.percentile(t2m[(np.isnan(t2m)==False) & (t2m<9999)],99.9), 20) 
		figinfo = [(8,5),levels,0,0,'NOAA Forecast, Temperature at 2 meters  '+str(gmtime(ntime)[2]).zfill(2)+'/'+str(gmtime(ntime)[1]).zfill(2)+'/'+repr(gmtime(ntime)[0])+' '+str(gmtime(ntime)[3]).zfill(2)+'Z'] 
		farr=forecastools.plotforecasts(t2m,lat,lon,figinfo) 

	version 1.1:    15/10/2015
	version 1.2:    08/04/2016
	version 1.3:    01/09/2016
        www.atmosmarine.com
	'''


	if len(args) < 3:
		sys.exit(' ERROR! Insuficient input arguments. At least one matrix (lat X lon), latitude vector and longitude vector must be provided.')

	elif len(args) == 3:
		data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); figinfo=0  # 2D plot with no info about fig parameters
		mtype=1
	elif len(args) == 4: 

		if min(np.atleast_2d(args[0]).shape) > 1: # 2D plot

			if min(np.atleast_2d(args[1]).shape) > 1:  # U and V, fig info not informed
				uwnd=copy.copy(args[0]); vwnd=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=0
				if uwnd.shape != vwnd.shape:
					sys.exit(' ERROR! U and V sizes are different')

				data=np.sqrt((uwnd*uwnd)+(vwnd*vwnd)) # magnitude (in case of two vector components inputs)
				if data.shape[0] != lat.shape[0] or data.shape[1] != lon.shape[0]:
					sys.exit(' ERROR! matrix size is not matching lat and/or lon sizes')

				mtype=2
			else: # data, fig info informed
				data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); figinfo=copy.copy(args[3])
				if data.shape[0] != lat.shape[0] or data.shape[1] != lon.shape[0]:
					sys.exit(' ERROR! matrix size is not matching lat and/or lon sizes')

				mtype=1
				if figinfo != 0:				
					if len(figinfo)<5:
						sys.exit(' ERROR! Insuficient figinfo input arguments. 5 arguments inside figinfo must be provided.')

		else:  # 1D plot, fig info not informed
			data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); ntime=copy.copy(args[3]); figinfo=0

	elif len(args) == 5:

		if min(np.atleast_2d(args[0]).shape) > 1: # 2D plot
			uwnd=copy.copy(args[0]); vwnd=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=copy.copy(args[4])
			if uwnd.shape != vwnd.shape:
				sys.exit(' ERROR! U and V sizes are different')
			data=np.sqrt((uwnd*uwnd)+(vwnd*vwnd)) # magnitude (in case of two vector components inputs).
			if data.shape[0] != lat.shape[0] or data.shape[1] != lon.shape[0]:
				sys.exit(' ERROR! matrix size is not matching lat and/or lon sizes')
			mtype=2
			if figinfo != 0:				
				if len(figinfo)<5:
					sys.exit(' ERROR! Insuficient figinfo input arguments. 5 arguments inside figinfo must be provided.')

		else: # 1D plot
			data=copy.copy(args[0]); lat=copy.copy(args[1]); lon=copy.copy(args[2]); ntime=copy.copy(args[3]); figinfo=copy.copy(args[4])

	else:
		sys.exit(' ERROR! Too many input arguments. 3 to 5 inputs are necessary only. help(forecastools.plotforecast) for more information.')

	# mask values
	if abs(np.array(data)).max() > 9.99e+20:
		data[abs(data)>9.99e+20]=NaN

	if len(np.atleast_1d(data).shape)>2:
		sys.exit(' ERROR! Wrong input data shape. It must be a one or two-dimensional variable')
	elif len(np.atleast_1d(data).shape)==1:
		# 1D plot, time series plot
		if np.atleast_1d(lat).shape[0] > 1 or np.atleast_1d(lon).shape[0] > 1 or np.atleast_1d(data).shape[0] < 2 :
			sys.exit(' ERROR! Wrong input data shape. Input data is a 1D vector array (time series will be ploted)')
		elif ntime.shape[0] != data.shape[0]:
			sys.exit(' ERROR! Time and data array must have the same length.')
		else:

			if np.atleast_1d(figinfo).shape[0] < 5:
				figinfo=[0,0,0,0,'']

			if figinfo[0] != 0:
				fig=plt.figure(figsize=figinfo[0])
			else:
				fig=plt.figure()

			year=np.zeros(ntime.shape[0],'i'); month=np.zeros(ntime.shape[0],'i'); day=np.zeros(ntime.shape[0],'i'); hour=np.zeros(ntime.shape[0],'i'); minute=np.zeros(ntime.shape[0],'i')
			for i in range(0,ntime.shape[0]):
				year[i]=gmtime(ntime[i])[0]; month[i]=gmtime(ntime[i])[1]; day[i]=gmtime(ntime[i])[2]; hour[i]=gmtime(ntime[i])[3]; minute[i]=gmtime(ntime[i])[4];

			dates = np.array([date2num(datetime(yy,mm,dd,hh,mi)) for yy,mm,dd,hh,mi in zip(year,month,day,hour,minute)])
			ax = plt.subplot2grid((7,1), (0,0), rowspan=6); plt.xticks(rotation=45)
			color = '0.75'; ax.plot_date(dates,data,color); plt.grid()
			ax.set_xlim( dates[0], dates[-1] )
			ax.plot_date(dates,data,'ko')
			ax.xaxis.set_major_formatter( DateFormatter('%d-%m-%Y %HZ') )
			plt.ylabel(" ", fontsize=8); plt.xlabel("Time", fontsize=10);plt.title(figinfo[4],fontsize=12)
			for label in ax.get_xticklabels() + ax.get_yticklabels():
				label.set_fontsize(8)

	else:
		# 2D plot, fields plot
		# defining automatic figure parameters in case input is not provided
		if figinfo==0:
			# level for cbar 
			if data[(np.isnan(data)==False)].min() != np.percentile(data[(np.isnan(data)==False)],99.99):
				levels = np.linspace( data[(np.isnan(data)==False)].min(), np.percentile(data[(np.isnan(data)==False)],99.99), 20)
			else:
				levels = np.linspace( data[(np.isnan(data)==False)].min(), data[(np.isnan(data)==False)].max(), 20)
			figinfo = [0,levels,0,0,'']
		#elif figinfo[1]==0:
		elif np.atleast_1d(figinfo[1]).shape[0]==1:
			if data[(np.isnan(data)==False)].min() != np.percentile(data[(np.isnan(data)==False)],99.99):
				levels = np.linspace( data[(np.isnan(data)==False)].min(), np.percentile(data[(np.isnan(data)==False)],99.99), 20)
			else:
				levels = np.linspace( data[(np.isnan(data)==False)].min(), data[(np.isnan(data)==False)].max(), 20)
			figinfo[1] = levels

		# cbar string format
		if (max(figinfo[1])-min(figinfo[1])) > 20:
			nformat='%4.0f'
		elif (max(figinfo[1])-min(figinfo[1])) > 5:
			nformat='%3.1f'
		elif (max(figinfo[1])-min(figinfo[1])) > 0.999:
			nformat='%3.2f'
		elif (max(figinfo[1])-min(figinfo[1])) > 0.1:
			nformat='%3.3f'
		else:
			nformat='%3.4f'

		# opening figure with figure size in case it is informed  
		if figinfo[0] != 0:
			fig=plt.figure(figsize=figinfo[0])
		else:
			fig=plt.figure()

		if figinfo[1].min()<0. and  ( abs( abs(figinfo[1].max())-abs(figinfo[1].min()) ) / abs( abs(figinfo[1].max())+abs(figinfo[1].min()) ) ) < 0.05 :
			palette = plt.cm.RdBu_r; palette.set_bad('aqua', 10.0)
		else:
			palette = plt.cm.jet; palette.set_bad('aqua', 10.0)

		# Plots (http://matplotlib.org/basemap/users/mapsetup.html)
		# (1) GLOBAL -----------
		if (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())>350:

			if lon[np.isnan(lon)==False].max()>350 :
				if mtype==2:
					uwnd[:,:],alon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),uwnd[:,:],lon,start=False)
					vwnd[:,:],alon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),vwnd[:,:],lon,start=False)
					data[:,:],lon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),data[:,:],lon,start=False)
				else:
					data[:,:],lon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),data[:,:],lon,start=False)


			# Global Plot with Eckert IV Projection	        	
			map = Basemap(projection='eck4',lon_0 = 0, resolution = 'c')
			#if np.any(data[np.isnan(data)==True]) or (min(figinfo[1])>data[np.isnan(data)==False].min()-0.001) or (max(figinfo[1])<data[np.isnan(data)==False].max()+0.001):
			#	map.bluemarble(scale=0.2); map.drawlsmask(ocean_color='w',land_color='None')
			#else:
			#	map.drawlsmask(ocean_color='w',land_color='grey') # map.fillcontinents(color='grey')
			map.drawlsmask(ocean_color='w',land_color='grey')
			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.contourf(xx,yy,data,figinfo[1],cmap=palette,extend="max")
			map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),40),labels=[0,0,0,1],linewidth=0.3,fontsize=6)
			map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),20),labels=[1,0,0,0],linewidth=0.3,fontsize=6)
			#map.drawstates(linewidth=0.1); #map.fillcontinents(color='gray')
			ax = plt.gca()
			pos = ax.get_position()
			l, b, w, h = pos.bounds
			cax = plt.axes([l+0.07, b+0.01, w-0.15, 0.025]) # setup colorbar axes.
			cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat);# cbar.ax.tick_params(labelsize=fsmpb)
			tick_locator = ticker.MaxNLocator(nbins=8); cbar.locator = tick_locator; cbar.update_ticks()
			plt.axes(ax)  # make the original axes current again
			plt.title(figinfo[4])
			# if it is a vector field, as winds and currents:
			if mtype==2:
				# figure parameter (skip vector) in case it is not provided
				if figinfo[2] ==0: 
					sx = round(13* (float(data[:,:].shape[0])*0.8 / float(data[:,:].shape[1])),1);  sy = 13 - sx
					figinfo[2] = int( (  (7.*(data[:,:].shape[0]+data[:,:].shape[1]) )/ 1080 ) * 41./(sx*sy) ) 
					figinfo[3] = 800 * (data[np.isnan(data)==False].max()/30.)
				[mnlon,mnlat]=np.meshgrid(lon[::figinfo[2]],lat[::figinfo[2]])
				xx2, yy2 = map(mnlon,mnlat)
				fwdt=0.001
				Q = map.quiver(xx2,yy2,uwnd[::figinfo[2],::figinfo[2]],vwnd[::figinfo[2],::figinfo[2]],width=fwdt,scale=figinfo[3])

		# (2) Regional covering high latitudes -----------
		elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())>20 and (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())<70 and abs(lat[np.isnan(lat)==False]).max()>60 and (float(lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())/float(lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()))<3. and (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())<60:

			# size/shape of the figure
			nwidth=int( (5000000*(lon[-1]-lon[0]))/50 )
			nheight=int( (5000000*(lat[-1]-lat[0]))/40 )
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )	
			# Stereographic Projection ------------------
			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 50:
				fres='f'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 100:
				fres='h'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 500:
				fres='i'

			map = Basemap(width=nwidth, height=nheight, resolution=fres, projection='stere',
				lat_ts=lat[0], lat_0=((lat[-1]-lat[0])/2)+lat[0], lon_0=((lon[-1]-lon[0])/2)+lon[0]    )

			#if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100:
			#	map.bluemarble(scale=np.tanh(4000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')
			map.drawlsmask(ocean_color='w',land_color='grey')

			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.contourf(xx,yy,data[:,:],figinfo[1],cmap=palette,extend="max")
			# map.drawstates(); map.fillcontinents(color='gray')
			map.drawmeridians(np.arange(round(lon[0]),round(lon[-1]),lonmd),labels=[0,0,0,1],linewidth=0.3)
			map.drawparallels(np.arange(round(lat[0]),round(lat[-1]),latmd),labels=[1,0,0,0],linewidth=0.3)
			ax = plt.gca(); pos = ax.get_position()
			if abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.7:
				db=0.06
			elif abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.5:
				db=0.05
			else:
				db=0.03
			l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
			cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
			tick_locator = ticker.MaxNLocator(nbins=8); cbar.locator = tick_locator; cbar.update_ticks()
			plt.axes(ax)  # make the original axes current again
			plt.title(figinfo[4]) 
			# if it is a vector field, as winds and currents:
			if mtype==2:
				# figure parameter (skip vector) and scale in case it is not provided
				if figinfo[2] == 0:
					npts=(float(data[:,:].shape[0])*(float(data[:,:].shape[1])))
					if npts<300:
						figinfo[2] = 1
					else:
						figinfo[2] = int(pow(npts,0.3)/3)
				figinfo[3]=int(400+pow(npts,0.4))*(data[np.isnan(data)==False].max()/30.)
				# vector grid and plot with quiver 
				[mnlon,mnlat]=np.meshgrid(lon[::figinfo[2]],lat[::figinfo[2]])
				xx2, yy2 = map(mnlon,mnlat)
				fwdt=0.003
				Q = map.quiver(xx2,yy2,uwnd[::figinfo[2],::figinfo[2]],vwnd[::figinfo[2],::figinfo[2]],width=fwdt,scale=figinfo[3])

		# (3) Regional -----------
		else:
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )
			# fig parameters
			if lat[0] < 0 and lat[-1] <= 0:
				lat_0=-(abs(lat[-1])+abs(lat[0]))/2.0
			else:
				lat_0=(lat[0]+lat[-1])/2.0
	
			if lon[0] < 0 and lon[-1] <= 0:
				lon_0=-(abs(lon[-1])+abs(lon[0]))/2.0
			else:
				lon_0=(lon[0]+lon[-1])/2.0

			#   Regional, Equidistant Cylindrical Projection ------------------
			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 50:
				fres='f'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 100:
				fres='h'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 500:
				fres='i'

			map = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution=fres)

			#if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100 and lon[-1]<=180:
			#	map.bluemarble(scale=np.tanh(5000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')
				#map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)
				#map.fillcontinents(color='grey',lake_color='aqua')
			map.drawlsmask(ocean_color='w',land_color='grey')

			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)  
			map.contourf(xx,yy,data[:,:],figinfo[1],cmap=palette,extend="max")
			# map.drawstates(); # map.fillcontinents(color='gray')
			map.drawmeridians(np.arange(round(lon.min()),round(lon.max()),lonmd),labels=[0,0,0,1])
			map.drawparallels(np.arange(round(lat.min()),round(lat.max()),latmd),labels=[1,0,0,0])
			ax = plt.gca(); pos = ax.get_position(); 
			if abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.7:
				db=0.06
			elif abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.5:
				db=0.05
			else:
				db=0.03
			l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
			cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
			tick_locator = ticker.MaxNLocator(nbins=8); cbar.locator = tick_locator; cbar.update_ticks()
			plt.axes(ax)  # make the original axes current again
			plt.title(figinfo[4]) 
			# if it is a vector field, as winds and currents:
			if mtype==2:
				# figure parameter (skip vector) and scale in case it is not provided
				npts=(float(data[:,:].shape[0])*(float(data[:,:].shape[1])))
				if figinfo[2] ==0: 
					if npts<300:
						figinfo[2] = 1
					else:
						figinfo[2] = int(pow(npts,0.3)/3)
				figinfo[3]=int(400+pow(npts,0.4))*(data[np.isnan(data)==False].max()/30.)
				# vector grid and plot with quiver
				[mnlon,mnlat]=np.meshgrid(lon[::figinfo[2]],lat[::figinfo[2]])
				xx2, yy2 = map(mnlon,mnlat)
				fwdt=0.002
				Q = map.quiver(xx2,yy2,uwnd[::figinfo[2],::figinfo[2]],vwnd[::figinfo[2],::figinfo[2]],width=fwdt,scale=figinfo[3])


	if mtype==2:
		return map, xx, yy, figinfo, xx2, yy2, fwdt
	else:
		return map, xx, yy, figinfo

	del figinfo, fig




def uploadit(*args):
	'''
	Function to upload figures or files to a server via ftp
	uploadit(server,userid,password,server_dir,localfilepath,filename)

	6 inputs are requested:
	server: server name between ""
	userid: user id between ""
	password: user password between ""
	server_dir: server diretory where you want to send the file, between ""
	localfilepath: source file diretory path between ""
	filename: file name to be uploaded between ""

	Examples (you must fill variables with your server info otherwise it will not work):

		import forecastools
		server="myforecast.com.br"  # fill with your server address
		userid=" "  # fill with your user id
		password=" "  # fill with your password
		server_dir="/41202"  # fill with your server diretory
		localfilepath="/media/rmc/chico/"  # fill with your local file path
		filename="NameOfInterest.jpg"  # fill with your figure name
		forecastools.uploadit(server,userid,password,server_dir,localfilepath,filename)

		from time import gmtime, strftime, time
		import forecastools
		import os
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10]),00]
		nt=9; lonlat=[0,360,-90,90];
		[uwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads[2]='vgrd10m'
		[vwnd,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		figinfo = [(8,5),0,20,800,'Global Wind Speed (m/s) at 10m for tomorrow: '+str(gmtime(ntime[-1])[2]).zfill(2)+'/'+str(gmtime(ntime[-1])[1]).zfill(2)+'/'+repr(gmtime(ntime[-1])[0])+' '+str(gmtime(ntime[-1])[3]).zfill(2)+'Z']
		forecastools.plotforecast(uwnd[-1,:,:],vwnd[-1,:,:],lat,lon,figinfo)
	       	savefig('namefig.jpg', dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='jpg',transparent=False, bbox_inches=None, pad_inches=0.1)
		server="myforecast.com.br"  # fill with your server address
		userid=" "  # fill with your user id
		password=" "  # fill with your password
		server_dir="/41202"  # fill with your server diretory
		localfilepath=os.getcwd()
		filename="namefig.jpg"  # fill with your figure name
		forecastools.uploadit(server,userid,password,server_dir,localfilepath,filename)

	version 1.1:    19/10/2015
	version 1.2:    08/04/2016
        www.atmosmarine.com
	'''

	if len(args) < 6:
		sys.exit(' ERROR! Insuficient input arguments. 6 inputs must be entered: server, userid, password, server_dir, localfilepath and filename')

	server=copy.copy(args[0]); userid=copy.copy(args[1]); password=copy.copy(args[2]); server_dir=copy.copy(args[3]); localfilepath=copy.copy(args[4]); filename=copy.copy(args[5])

	# a simple solution in case localfilepath does not end with '/'
	if localfilepath[-1] != '/':
		localfilepath = localfilepath+'/'

	# a simple solution in case localfilepath does not end with '/'
	if server_dir[0] != '/':
		server_dir = '/'+server_dir

	ftp = ftplib.FTP(server)
	ftp.login(userid, password)
	ftp.cwd(server_dir)
	upload_file = open(localfilepath+filename, 'r')
	ftp.storbinary('STOR '+ filename, upload_file)


def list(*args):
	'''
	List of nomads OpenDAP variables

	import forecastools
	forecastools.list()

	version 1.1:	18/10/2015
	version 1.2:    08/04/2016
        www.atmosmarine.com
	'''

	print("	\n \
	Sea waves products link: \n \
	http://nomads.ncep.noaa.gov:9090/dods/wave/ \n \
	Mostly used variables: \n \
	  'dirpwsfc': primary wave direction [deg] \n \
	  'dirswsfc': secondary wave direction [deg] \n \
	  'htsgwsfc': significant height of combined wind waves and swell [m] \n \
	  'perpwsfc': primary wave mean period [s] \n \
	  'perswsfc': secondary wave mean period [s] \n \
	  'ugrdsfc': u-component of wind [m/s] \n \
	  'vgrdsfc': v-component of wind [m/s] \n \
	  'wdirsfc': wind direction (from which blowing) [deg] \n \
	  'windsfc': wind speed [m/s] \n \
	  'wvdirsfc': direction of wind waves [deg] \n \
	  'wvpersfc': mean period of wind waves [s] \n \
	\n \
	 Atmospheric products link: \n \
	 GFS 0.25:  	 http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/ \n \
	 Mostly used variables: \n \
	  'ugrd10m': 10 m above ground u-component of wind [m/s] \n \
	  'vgrd10m': 10 m above ground v-component of wind [m/s] \n \
	  'ugrdprs': (1000 975 950 925 900.. 70 50 30 20 10) u-component of wind [m/s] \n \
	  'vgrdprs': (1000 975 950 925 900.. 70 50 30 20 10) v-component of wind [m/s]  \n \
	  'gustsfc': surface wind speed (gust) [m/s] \n \
	  'ugrdmwl': max wind u-component of wind [m/s] \n \
	  'vgrdmwl': max wind v-component of wind [m/s] \n \
	  'uflxsfc': surface momentum flux, u-component [n/m^2] \n \
	  'vflxsfc': surface momentum flux, v-component [n/m^2] \n \
	  'ugrd2pv': pv=2e-06 (km^2/kg/s) surface u-component of wind [m/s] \n \
	  'vgrd2pv': pv=2e-06 (km^2/kg/s) surface v-component of wind [m/s] \n \
	  'absvprs': (1000 975 950 925 900.. 70 50 30 20 10) absolute vorticity [1/s] \n \
	  'hgtprs': (1000 975 950 925 900.. 70 50 30 20 10) geopotential height [gpm] \n \
	\n \
	  'tmax2m' : 2 m above ground maximum temperature [k] \n \
	  'tmin2m' : 2 m above ground minimum temperature [k] \n \
	  'tmpsfc' : surface temperature [k] \n \
	  'tmp2m' :  2 m above ground temperature [k] \n \
	\n \
	  'pressfc': surface pressure [pa] \n \
	  'prmslmsl': mean sea level pressure reduced to msl [pa] \n \
	\n \
	  'absvprs': absolute vorticity [1/s] \n \
	  'no5wavh500mb':  500 mb 5-wave geopotential height [gpm] \n \
	  'hgtsfc': surface geopotential height [gpm] \n \
	\n \
	  'icecsfc': surface ice cover [proportion] \n \
	  'landsfc': surface land cover (0=sea, 1=land) [proportion]\n \
	\n \
	  'acpcpsfc': surface convective precipitation [kg/m^2] \n \
	  'apcpsfc': surface total precipitation [kg/m^2] \n \
	  'clwmrprs': (1000 975 950 925 900.. 300 250 200 150 100) cloud mixing ratio [kg/kg] \n \
	  'cpratsfc': surface convective precipitation rate [kg/m^2/s] \n \
	  'crainsfc': surface categorical rain [-] \n \
	  'cwatclm': entire atmosphere (considered as a single layer) cloud water [kg/m^2] \n \
	  'cworkclm': entire atmosphere (considered as a single layer) cloud work function [j/kg] \n \
	  'pratesfc': surface precipitation rate [kg/m^2/s] \n \
	  'pwatclm': entire atmosphere (considered as a single layer) precipitable water [kg/m^2] \n \
	  'rhprs': (1000 975 950 925 900.. 70 50 30 20 10) relative humidity [%] \n \
	  'rhclm': entire atmosphere (considered as a single layer) relative humidity [%] \n \
	  'rh2m': 2 m above ground relative humidity [%] \n \
	  'spfh2m': 2 m above ground specific humidity [kg/kg] \n \
	  'tcdcclm': entire atmosphere total cloud cover [%] \n \
	  'tcdclcll': low cloud layer total cloud cover [%] \n \
	  'tcdcmcll': middle cloud layer total cloud cover [%] \n \
	  'tcdchcll': high cloud layer total cloud cover [%] \n \
	  'tcdcccll': convective cloud layer total cloud cover [%] \n \
	\n \
	 Global Real-Time Ocean Forecast System (2D only): \n \
	 HYCOM (HYbrid Coordinates Ocean Model) 1/12 degree :  http://nomads.ncep.noaa.gov:9090/dods/rtofs/ \n \
	 Mostly used variables: \n \
	  'u_velocity':   eastward_sea_water_velocity (m/s) \n \
	  'v_velocity':   northward_sea_water_velocity (m/s) \n \
	  'sst':  sea_surface_temperature (c) \n \
	  'sss':  sea_surface_salinity (kg/kg)")


def distance(*args):
	'''
	Function to calculate distance (in meters) between two point (lat/lon) in Spherical Coordinates.

	4 inputs are requested:
	lat1: initial latitude
	lon1: initial longitude
	lat2: final latitude
	lon2: final longitude

	1 output is provided;
	distance : distance (in meters) between two points

	d=arccos(sin(lat1*pi/180.)*sin(lat2*pi/180.) + cos(lat1*pi/180.)*cos(lat2*pi/180.)*cos(lon2*pi/180.-lon1*pi/180.) ) * 6371000.

	Examples:

	import forecastools
	lat1=38.7139; lon1=-9.1394
	lat2=38.9047; lon2=-77.0164
	fdist=forecastools.distance(lat1,lon1,lat2,lon2)

	version 1.1:    25/10/2015
	version 1.2:    08/04/2016
        www.atmosmarine.com
	'''

	if len(args) < 4:
		sys.exit(' ERROR! Insuficient input arguments. 4 inputs must be entered: initial latitude, initial longitude, final latitude, final longitude')
	else:
		lat1=copy.copy(args[0]); lon1=copy.copy(args[1]); lat2=copy.copy(args[2]); lon2=copy.copy(args[3]);

	if lon1<0:
		lon1=lon1+360
	if lon2<0:
		lon2=lon2+360

	fdist=np.arccos(np.sin(lat1*np.pi/180.)*np.sin(lat2*np.pi/180.) + np.cos(lat1*np.pi/180.)*np.cos(lat2*np.pi/180.)*np.cos(lon2*np.pi/180.-lon1*np.pi/180.) ) * 6371000.
	return round(fdist,2)



# next two small functions are used to plot circles with defined radius (km)

def shoot(lon, lat, azimuth, maxdist=None):
	"""Shooter Function
	Original javascript on http://williams.best.vwh.net/gccalc.htm
	Translated to python by Thomas Lecocq
	"""
	glat1 = lat * np.pi / 180.
	glon1 = lon * np.pi / 180.
	s = maxdist / 1.852
	faz = azimuth * np.pi / 180.
 
	EPS= 0.00000000005
	if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
		alert("Only N-S courses are meaningful, starting at a pole!")
 
	a=6378.13/1.852
	f=1/298.257223563
	r = 1 - f
	tu = r * np.tan(glat1)
	sf = np.sin(faz)
	cf = np.cos(faz)
	if (cf==0):
		b=0.
	else:
		b=2. * np.arctan2 (tu, cf)
 
	cu = 1. / np.sqrt(1 + tu * tu)
	su = tu * cu
	sa = cu * sf
	c2a = 1 - sa * sa
	x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
	x = (x - 2.) / x
	c = 1. - x
	c = (x * x / 4. + 1.) / c
	d = (0.375 * x * x - 1.) * x
	tu = s / (r * a * c)
	y = tu
	c = y + 1
	while (np.abs (y - c) > EPS):
 
		sy = np.sin(y)
		cy = np.cos(y)
		cz = np.cos(b + y)
		e = 2. * cz * cz - 1.
		c = y
		x = e * cy
		y = e + e - 1.
		y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) * d / 4. - cz) * sy * d + tu
 
	b = cu * cy * cf - su * sy
	c = r * np.sqrt(sa * sa + b * b)
	d = su * cy + cu * sy * cf
	glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
	c = cu * cy - su * sy * cf
	x = np.arctan2(sy * sf, c)
	c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
	d = ((e * cy * c + cz) * sy * c + y) * sa
	glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi    

	baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)
 
	glon2 *= 180./np.pi
	glat2 *= 180./np.pi
	baz *= 180./np.pi
 
	return (glon2, glat2, baz)
 
def equi(m, centerlon, centerlat, radius, cor, *args, **kwargs):
	'''
	Function to plot circles with a defined radius (km)
	Example:
	   fig = plt.figure(figsize=(11.7,8.3)) 
	   #Custom adjust of the subplots
	   plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
	   ax = plt.subplot(111)
	   #Let's create a basemap of the world
	   m = Basemap(resolution='l',projection='robin',lon_0=0)
	   m.drawcountries()
	   m.drawcoastlines()
	   m.fillcontinents(color='grey',lake_color='white')
	   m.drawparallels(np.arange(-90.,120.,30.))
	   m.drawmeridians(np.arange(0.,360.,60.))
	   m.drawmapboundary(fill_color='white') 
	   centerlon = 0.; centerlat = 0.
	   radius=200.
	   forecastools.equi(m, centerlon, centerlat, radius,'k',lw=1.)
	'''

	glon1 = centerlon
	glat1 = centerlat
	X = []
	Y = []
	for azimuth in range(0, 360):
		glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
		X.append(glon2)
		Y.append(glat2)
	X.append(X[0])
	Y.append(Y[0])

	X,Y = m(X,Y)
	plt.plot(X,Y,cor,**kwargs)
 



