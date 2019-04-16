#!/usr/bin/python
#
# Cyclone Tracker
#
# Group of functions to work with cyclones. Examples are using NCEP/NOAA 
# forecast fields but it accepts any input.
# The cyclone identification and tracking is strongly based on the method 
# of NCEP/NOAA: "HOW THE NCEP TROPICAL CYCLONE TRACKER WORKS", by Timothy P. Marchok
#
# The python module forecastools.py is required and should have been 
# provided together with this code.
#
# Dependencies
# See the few lines bellow to verify python dependencies. An easy way to install it is by using python anaconda:
# https://store.continuum.io/cshop/anaconda/
# which can be installed without administration permission. After installation and after adding anaconda path to .bashrc or equivalent:
# conda install numpy
# conda install pylab
# conda install basemap
# conda install matplotlib
# 
# Functions defined below:
# cyclone.findcandidates    # To first identify cyclone candidates (independent in time) from meteorological input fields. This function might require a long time of running if large areas with fine grids are used.
# cyclone.position          # Calculate cyclone positions using candidates previously identified and it tests events with additional imposed restrictions.
# cyclone.linktime          # Take independent cyclone positions calculated with cyclone.position and link cyclones in time, creating tracks and cyclone evolution.
# cyclone.ctable            # Create a cyclone table with min, max and mean values of an input variable(data) within each cyclone.
# cyclone.cplot             # Plot cyclone tracks using Basemap. All tracks are plotted together and a background field (additional input matrix) can also be entered.
# cyclone.cplotic           # Plot cyclone tracks using Basemap. One figure per cyclone. A background field (additional input matrix) can also be entered.
# cyclone.cplotbm           # Plot cyclone tracks using Basemap. All tracks are plotted together on a bluemarble Basemap field. Little information about the cyclones are shown, only the tracks plotted in red.
#
# Functions findcandidates, position and linktime are used for the cyclone indentification and tracking, divided in three steps. See the manual for a detailed description.
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
#	version 1.1:    03/11/2015
#	version 1.2:    08/04/2016
#	version 1.3:    08/05/2016
#	version 1.4:    22/07/2016
#     version 1.5: 
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Contributions: Anaconda Community (https://www.continuum.io/anaconda-community), MotorDePopa Wave Research Group (motordepopa@googlegroups.com) 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from pylab import *
import copy
from time import gmtime
from mpl_toolkits.basemap import Basemap, shiftgrid
from matplotlib.gridspec import GridSpec
from matplotlib.mlab import *
from matplotlib import ticker
from matplotlib import *
import matplotlib
import timeit
import sys
import numpy as np
from calendar import timegm
import time
import forecastools

palette = plt.cm.jet
palette.set_bad('aqua', 10.0)


def findcandidates(*args):
	'''
 	1) Find cyclone candidates (independent in time) strongly based on the method of NCEP/NOAA:
	"HOW THE NCEP TROPICAL CYCLONE TRACKER WORKS", Timothy P. Marchok
	which uses the mean sea level pressure, vorticity and geopotential height to initially identify the cyclone and its position.
	findcandidates is the first step of a group of three functions of cyclone tracking

	candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)

	7 inputs are requested:
	lat : vector of latitudes, len(lat.shape) must be one.
	lon : vector of longitudes, len(lon.shape) must be one.
	prmslmsl : mean sea level pressure (one z level only). prmslmsl.shape  is  time, latitudes and longitudes
	absvprs : absolute vorticity at 850mb and 700mb (two z levels). absvprs.shape is time, levels, latitudes and longitudes. levels must be 850 and 700 mb.
	hgtprs : geopotential height at 850mb and 700mb (two z levels). hgtprs.shape is time, levels, latitudes and longitudes. levels must be 850 and 700 mb.
	threshold : vector of thresholds (minimum or maximum value) associated with criteria of identification of each parameter (pressure, vorticity, geopotential etc);
                   threshold[0] maximum mean sea level pressure
		   threshold[1] minimum 850mb absolute vorticity
		   threshold[2] minimum 700mb absolute vorticity
		   threshold[3] maximum 850mb geopotential height
		   threshold[4] maximum 700mb geopotential height
		For Ex: threshold=[1010.,0.0004,0.0003,1300.,3000.]
		The minimum/maximum threshold is turned off when entering zero; i.e., the cyclone is identified based on local minimum/maximum without restriction of values.
	sw : vector containing the size (km) of the sub-grid that will run the entire grid/fields searching for cyclones. No more than one cyclone will be identified within the same sub-grid.
		sw has size-in-latitude and size-in-longitude respectively
		For Ex: sw=np.array([275.,275.]) # 275 km is used by NCEP for tropical cyclones

	User does not need to use all fields of prmslmsl, absvprs and hgtprs. You can set as zero and the varible will be turned off.
	Obviously, at least one variable/field must be entered. We recommend to use at least the mean sea level pressure for the identification.

	1 output is provided;
	candidates : matrix with zero or one, with shape of variable/field, time, latitude, longitude.
		    0: zero means no cyclone found
                    1: one is the cyclone candidate position found.
		    NaN: variable ignored
		    Therefore, a cyclone can be found in one parameter (prmslmsl for example) but not in the other. This is why it is called candidate.
		candidates.shape[0] will always be 5, even if you have not entered one variable. In this case candidates will contain NaN instead of 0 or 1.

	Be careful in order to not be over-restrictive. Light criterias together can lead to very restrictive conditions and important cyclones can be missed.
	Cyclones are very different over the globe and especially over different latitudes. Define certain thresholds for a family of cyclones within a defined region and different threshold for other cases.

	Example:

		from pylab import * 
		from time import gmtime, strftime, time 
		import forecastools 
		import cyclone 
		lonlat=[-80,20,-80,80]; nt=57; 
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()) # today 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])] 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p50','prmslmsl'] 
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat) 
		levels=[850,700] 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p50','absvprs'] 
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels) 
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p50','hgtprs'] 
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels) 
		swlat=275.; swlon=275.; sw=np.array([swlat,swlon]) 
		threshold=[1020.,5./100000.,2./100000.,1500.,3050.] 
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		t=10 
		forecastools.plotforecast(candidates[0,t,:,:],lat,lon) # show points of cyclone candidates from mean sea level pressure field at instant t. You can play with that to visualize other times and variables 
		forecastools.plotforecast(candidates[1,t,:,:],lat,lon) # absolute vorticity [1/s] at 850 mb
		forecastools.plotforecast(candidates[2,t,:,:],lat,lon) # absolute vorticity [1/s] at 700 mb
		forecastools.plotforecast(candidates[3,t,:,:],lat,lon) # geopotential height [geopotential meter, gpm] at 850 mb
		forecastools.plotforecast(candidates[4,t,:,:],lat,lon) # geopotential height [geopotential meter, gpm] at 700 mb


	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	start = timeit.default_timer()

	if len(args) < 7:
		sys.exit(' ERROR! Insufficient input arguments. lat lon vectors; prmslmsl, absvprs and hgtprs matrices and threshold and sw vectors must be entered ')
	elif len(args) == 7:
		lat=copy.copy(args[0]); lon=copy.copy(args[1]); prmslmsl=copy.copy(args[2]); absvprs=copy.copy(args[3]); hgtprs=copy.copy(args[4]); threshold=copy.copy(args[5]); 
		if len(args[6])==2:
			swlat=copy.copy(args[6][0]); swlon=copy.copy(args[6][1])
		else:
			sys.exit(' ERROR! Problems with shape. Sw must have two values related to swlat and swlon. For ex: sw=np.array([275,275])')
	if len(lat.shape)>1 or len(lon.shape)>1:
		sys.exit(' ERROR! Problems with shape. Latitude and Longitude must be the first inputs')

	tt=[]
	if np.mean(prmslmsl)!=0:
		if len(prmslmsl.shape)!=3 :
			sys.exit(' ERROR! Problems with shape. It must be: len(prmslmsl.shape)=3')
		else:
			tt=np.append(tt,prmslmsl.shape[0])
	else:
		print(' Warning! You should not turn off the sea level pressure fields criteria! It is one of the most important variables for the cyclone identification')

	if np.mean(absvprs)!=0:
		if len(absvprs.shape)!=4 or absvprs.shape[1]!=2:
			sys.exit(' ERROR! Problems with shape. It must be: len(absvprs.shape)=4; i.e., prmslmsl has only one Z level (surface) and absvprs and hgtprs have two')
		else:
			tt=np.append(tt,absvprs.shape[0])
	if np.mean(hgtprs)!=0:
		if len(hgtprs.shape)!=4 or hgtprs.shape[1]!=2:
			sys.exit(' ERROR! Problems with shape. It must be: len(hgtprs.shape)=4; i.e., prmslmsl has only one Z level (surface) and absvprs and hgtprs have two')
		else:
			tt=np.append(tt,hgtprs.shape[0])

	if len(tt)==0:
		sys.exit(' ERROR! At least one parameter field (prmslmsl, absvprs or hgtprs) must be entered')
	elif abs(np.diff(tt)).mean()>0:
		sys.exit(' ERROR! Problems with shape. Parameters/fields must have the same time and size')
	else:
		tt=int(np.mean(tt))

	# Turn off threshold criteria when entered zero
	if len(threshold)!=5:
		if threshold == 0:
			threshold=[999999.,0.,0.,999999.,999999.]
		else:
			sys.exit(' ERROR! Problems with shape. Threshold must have 5 inputs')
	else:
		if threshold[0]==0:
			threshold[0]=999999.
		if threshold[3]==0:
			threshold[3]=999999.
		if threshold[4]==0:
			threshold[4]=999999.
	threshold = np.copy(np.array(threshold,'f'))

	# Converting pressure to mb
	if np.mean(prmslmsl) > 95000:
		prmslmsl=np.copy(prmslmsl/100.)
	# searching sub-grid in meters
	swlat=swlat*1000.; swlon=swlon*1000.

	# From Cartesian to Spherical coordinates , distance in meters 
	latdist=np.zeros((lat.shape[0]),'f') # lat distance of grid points
	londist=np.zeros((lat.shape[0],lon.shape[0]),'f') # lon distance of grid points
	for i in range(0,lat.shape[0]):
		latdist[i]=forecastools.distance(lat[0],lon[0],lat[i],lon[0])
		for j in range(1,lon.shape[0]):
			londist[i,j]=londist[i,j-1]+forecastools.distance(lat[i],lon[j-1],lat[i],lon[j])
	londist[np.isnan(londist)==True]=0; latdist[np.isnan(latdist)==True]=0

	# initial and final index defining the searching area
	ilati=find(latdist>swlat/2.)[0]
	ilatf=find( abs(latdist-latdist.max()) > swlat/2.)[-1]
	if np.diff(latdist[ilati:ilatf+1]).sum() < swlat :
		sys.exit(' ERROR! Area not large enough for the algorithm (latitude) or swlat too big')
	iloni=find(londist[np.where(abs(lat)==abs(lat).min())[0],:][0]>swlat/2.)[0]
	ilonf=find( abs(londist[np.where(abs(lat)==abs(lat).min())[0],:][0]-londist[np.where(abs(lat)==abs(lat).min())[0],:][0].max()) > swlon/2.)[-1]
	if np.diff(londist[0,:]).sum() < swlon or np.diff(londist[-1,:]).sum() < swlon:
		sys.exit(' ERROR! Area not large enough for the algorithm (longitude) or swlon too big')

	# matrix of cyclone candidates (found and approved) related to each variable. 
	candidates = np.zeros( (5,tt,lat.shape[0],lon.shape[0]), 'f') 
	cindx=np.zeros((1,7),'f') # indexes of the cyclone positions

	# Check if detection sub-grid fit into the input grid (spherical coordinates), for latitude 
	ai = int(np.where( latdist > (swlat/2) )[0][0])
	bi = int(np.where( abs(latdist-latdist[-1]) > (swlat/2) )[0][-1])
	# Check if detection sub-grid fit into the input grid (spherical coordinates), for longitude
	aj=np.zeros((londist.shape[0]),'f'); bj=np.zeros((londist.shape[0]),'f');
	for i in range(0,londist.shape[0]):
		aj[i] = int(np.where( londist[i,:] > (swlon/2) )[0][0]) 
		bj[i] = int(np.where( abs(londist[i,:] - londist[i,-1]) > (swlon/2) )[0][-1]) 

	print('cyclone.findcandidates:  grids and conditions initialized')
	# loop through grid points
	if np.mean(prmslmsl) != 0 and np.mean(absvprs) != 0 and np.mean(hgtprs) != 0:
		for i in range(int(ai),int(bi)):
			for j in range(int(aj[i]),int(bj[i])):
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon
				# loop in time
				for t in range(0,int(tt)):
					# NCEP/NOAA identification method: track based on an average of the positions of 5 different primary parameters:
					#       (mslp, 700 and 850 mb relative vorticity, 700 and 850 mb geopotential height)
					# 1) mean sea level Pressure
					if round(prmslmsl[t,i,j],3) == round(prmslmsl[t,ii:fi+1,ij:fj+1].min(),3) and prmslmsl[t,i,j] <= threshold[0]:
						candidates[0,t,i,j]=1
					# 2) Vorticity
					# 2.1) 850mb absolute vorticity
					if lat[i] > 0:  # North Hemisphere, positive vorticity
						if round(absvprs[t,0,i,j],5) == round(absvprs[t,0,ii:fi+1,ij:fj+1].max(),5) and absvprs[t,0,i,j] >= abs(threshold[1]):
							candidates[1,t,i,j]=1
					else:  # South Hemisphere, negative vorticity
						if round(absvprs[t,0,i,j],5) == round(absvprs[t,0,ii:fi+1,ij:fj+1].min(),5) and absvprs[t,0,i,j] <= -abs(threshold[1]):
							candidates[1,t,i,j]=1
					# 2.2) 700mb absolute vorticity
					if lat[i] > 0:   # North Hemisphere, positive vorticity
						if round(absvprs[t,1,i,j],5) == round(absvprs[t,1,ii:fi+1,ij:fj+1].max(),5) and absvprs[t,1,i,j] >= abs(threshold[2]):
							candidates[2,t,i,j]=1
					else:   # South Hemisphere, negative vorticity
						if round(absvprs[t,1,i,j],5) == round(absvprs[t,1,ii:fi+1,ij:fj+1].min(),5) and absvprs[t,1,i,j] <= -abs(threshold[2]):
							candidates[2,t,i,j]=1
					# 3) Geopotential
					# 3.1) 850mb geopotential height
					if round(hgtprs[t,0,i,j],3) == round(hgtprs[t,0,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,0,i,j] <= threshold[3]:
						candidates[3,t,i,j]=1
					# 3.2) 700mb geopotential height
					if round(hgtprs[t,1,i,j],3) == round(hgtprs[t,1,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,1,i,j] <= threshold[4]:
						candidates[4,t,i,j]=1
					# After detection and tests, save the position (including sub-grid) and time
					if sum(candidates[(np.isnan(candidates[:,t,i,j])==False),t,i,j]) > 0 :
						cindx=np.append(cindx,[[int(ii),int(i),int(fi),int(ij),int(j),int(fj),int(t)]],axis=0)
			print('cyclone.findcandidates:  grid loop, '+repr(i-ai+1)+'/'+repr(bi-ai)+'  done...')


	if np.mean(prmslmsl) == 0 and np.mean(absvprs) != 0 and np.mean(hgtprs) != 0:
		candidates[0,:,:,:]=np.nan
		for i in range(int(ai),int(bi)):
			for j in range(int(aj[i]),int(bj[i])):
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon
				# loop in time
				for t in range(0,int(tt)):
					# NCEP/NOAA identification method: track based on an average of the positions of 5 different primary parameters:
					#       (mslp, 700 and 850 mb relative vorticity, 700 and 850 mb geopotential height)
					# 2) Vorticity
					# 2.1) 850mb absolute vorticity
					if lat[i] > 0:  # North Hemisphere, positive vorticity
						if round(absvprs[t,0,i,j],5) == round(absvprs[t,0,ii:fi+1,ij:fj+1].max(),5) and absvprs[t,0,i,j] >= abs(threshold[1]):
							candidates[1,t,i,j]=1
					else:  # South Hemisphere, negative vorticity
						if round(absvprs[t,0,i,j],5) == round(absvprs[t,0,ii:fi+1,ij:fj+1].min(),5) and absvprs[t,0,i,j] <= -abs(threshold[1]):
							candidates[1,t,i,j]=1
					# 2.2) 700mb absolute vorticity
					if lat[i] > 0:   # North Hemisphere, positive vorticity
						if round(absvprs[t,1,i,j],5) == round(absvprs[t,1,ii:fi+1,ij:fj+1].max(),5) and absvprs[t,1,i,j] >= abs(threshold[2]):
							candidates[2,t,i,j]=1
					else:   # South Hemisphere, negative vorticity
						if round(absvprs[t,1,i,j],5) == round(absvprs[t,1,ii:fi+1,ij:fj+1].min(),5) and absvprs[t,1,i,j] <= -abs(threshold[2]):
							candidates[2,t,i,j]=1
					# 3) Geopotential
					# 3.1) 850mb geopotential height
					if round(hgtprs[t,0,i,j],3) == round(hgtprs[t,0,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,0,i,j] <= threshold[3]:
						candidates[3,t,i,j]=1
					# 3.2) 700mb geopotential height
					if round(hgtprs[t,1,i,j],3) == round(hgtprs[t,1,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,1,i,j] <= threshold[4]:
						candidates[4,t,i,j]=1
					# After detection and tests, save the position (including sub-grid) and time
					if sum(candidates[(np.isnan(candidates[:,t,i,j])==False),t,i,j]) > 0 :
						cindx=np.append(cindx,[[int(ii),int(i),int(fi),int(ij),int(j),int(fj),int(t)]],axis=0)
			print('cyclone.findcandidates:  grid loop, '+repr(i-ai+1)+'/'+repr(bi-ai)+'  done...')


	if np.mean(prmslmsl) == 0 and np.mean(absvprs) == 0 and np.mean(hgtprs) != 0:
		candidates[0,:,:,:]=np.nan; candidates[1,:,:,:]=np.nan; candidates[2,:,:,:]=np.nan
		for i in range(int(ai),int(bi)):
			for j in range(int(aj[i]),int(bj[i])):
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon
				# loop in time
				for t in range(0,int(tt)):
					# NCEP/NOAA identification method: track based on an average of the positions of 5 different primary parameters:
					#       (mslp, 700 and 850 mb relative vorticity, 700 and 850 mb geopotential height)
					# 3) Geopotential
					# 3.1) 850mb geopotential height
					if round(hgtprs[t,0,i,j],3) == round(hgtprs[t,0,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,0,i,j] <= threshold[3]:
						candidates[3,t,i,j]=1
					# 3.2) 700mb geopotential height
					if round(hgtprs[t,1,i,j],3) == round(hgtprs[t,1,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,1,i,j] <= threshold[4]:
						candidates[4,t,i,j]=1
					# After detection and tests, save the position (including sub-grid) and time
					if sum(candidates[(np.isnan(candidates[:,t,i,j])==False),t,i,j]) > 0 :
						cindx=np.append(cindx,[[int(ii),int(i),int(fi),int(ij),int(j),int(fj),int(t)]],axis=0)
			print('cyclone.findcandidates:  grid loop, '+repr(i-ai+1)+'/'+repr(bi-ai)+'  done...')


	if np.mean(prmslmsl) != 0 and np.mean(absvprs) != 0 and np.mean(hgtprs) == 0:
		candidates[3,:,:,:]=np.nan; candidates[4,:,:,:]=np.nan
		for i in range(int(ai),int(bi)):
			for j in range(int(aj[i]),int(bj[i])):
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon
				# loop in time
				for t in range(0,int(tt)):
					# NCEP/NOAA identification method: track based on an average of the positions of 5 different primary parameters:
					#       (mslp, 700 and 850 mb relative vorticity, 700 and 850 mb geopotential height)
					# 1) mean sea level Pressure
					if round(prmslmsl[t,i,j],3) == round(prmslmsl[t,ii:fi+1,ij:fj+1].min(),3) and prmslmsl[t,i,j] <= threshold[0]:
						candidates[0,t,i,j]=1
					# 2) Vorticity
					# 2.1) 850mb absolute vorticity
					if lat[i] > 0:  # North Hemisphere, positive vorticity
						if round(absvprs[t,0,i,j],5) == round(absvprs[t,0,ii:fi+1,ij:fj+1].max(),5) and absvprs[t,0,i,j] >= abs(threshold[1]):
							candidates[1,t,i,j]=1
					else:  # South Hemisphere, negative vorticity
						if round(absvprs[t,0,i,j],5) == round(absvprs[t,0,ii:fi+1,ij:fj+1].min(),5) and absvprs[t,0,i,j] <= -abs(threshold[1]):
							candidates[1,t,i,j]=1
					# 2.2) 700mb absolute vorticity
					if lat[i] > 0:   # North Hemisphere, positive vorticity
						if round(absvprs[t,1,i,j],5) == round(absvprs[t,1,ii:fi+1,ij:fj+1].max(),5) and absvprs[t,1,i,j] >= abs(threshold[2]):
							candidates[2,t,i,j]=1
					else:   # South Hemisphere, negative vorticity
						if round(absvprs[t,1,i,j],5) == round(absvprs[t,1,ii:fi+1,ij:fj+1].min(),5) and absvprs[t,1,i,j] <= -abs(threshold[2]):
							candidates[2,t,i,j]=1
					# After detection and tests, save the position (including sub-grid) and time
					if sum(candidates[(np.isnan(candidates[:,t,i,j])==False),t,i,j]) > 0 :
						cindx=np.append(cindx,[[int(ii),int(i),int(fi),int(ij),int(j),int(fj),int(t)]],axis=0)
			print('cyclone.findcandidates:  grid loop, '+repr(i-ai+1)+'/'+repr(bi-ai)+'  done...')


	if np.mean(prmslmsl) != 0 and np.mean(absvprs) == 0 and np.mean(hgtprs) != 0:
		candidates[1,:,:,:]=np.nan; candidates[2,:,:,:]=np.nan
		for i in range(int(ai),int(bi)):
			for j in range(int(aj[i]),int(bj[i])):
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon
				# loop in time
				for t in range(0,int(tt)):
					# NCEP/NOAA identification method: track based on an average of the positions of 5 different primary parameters:
					#       (mslp, 700 and 850 mb relative vorticity, 700 and 850 mb geopotential height)
					# 1) mean sea level Pressure
					if round(prmslmsl[t,i,j],3) == round(prmslmsl[t,ii:fi+1,ij:fj+1].min(),3) and prmslmsl[t,i,j] <= threshold[0]:
						candidates[0,t,i,j]=1
					# 3) Geopotential
					# 3.1) 850mb geopotential height
					if round(hgtprs[t,0,i,j],3) == round(hgtprs[t,0,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,0,i,j] <= threshold[3]:
						candidates[3,t,i,j]=1
					# 3.2) 700mb geopotential height
					if round(hgtprs[t,1,i,j],3) == round(hgtprs[t,1,ii:fi+1,ij:fj+1].min(),3) and hgtprs[t,1,i,j] <= threshold[4]:
						candidates[4,t,i,j]=1
					# After detection and tests, save the position (including sub-grid) and time
					if sum(candidates[(np.isnan(candidates[:,t,i,j])==False),t,i,j]) > 0 :
						cindx=np.append(cindx,[[int(ii),int(i),int(fi),int(ij),int(j),int(fj),int(t)]],axis=0)
			print('cyclone.findcandidates:  grid loop, '+repr(i-ai+1)+'/'+repr(bi-ai)+'  done...')


	if np.mean(prmslmsl) != 0 and np.mean(absvprs) == 0 and np.mean(hgtprs) == 0:
		candidates[1,:,:,:]=np.nan; candidates[2,:,:,:]=np.nan; candidates[3,:,:,:]=np.nan; candidates[4,:,:,:]=np.nan
		for i in range(int(ai),int(bi)):
			for j in range(int(aj[i]),int(bj[i])):
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon
				# loop in time
				for t in range(0,int(tt)):
					# NCEP/NOAA identification method: track based on an average of the positions of 5 different primary parameters:
					#       (mslp, 700 and 850 mb relative vorticity, 700 and 850 mb geopotential height)
					# 1) mean sea level Pressure
					if round(prmslmsl[t,i,j],3) == round(prmslmsl[t,ii:fi+1,ij:fj+1].min(),3) and prmslmsl[t,i,j] <= threshold[0]:
						candidates[0,t,i,j]=1
					# After detection and tests, save the position (including sub-grid) and time
					if sum(candidates[(np.isnan(candidates[:,t,i,j])==False),t,i,j]) > 0 :
						cindx=np.append(cindx,[[int(ii),int(i),int(fi),int(ij),int(j),int(fj),int(t)]],axis=0)
			print('cyclone.findcandidates:  grid loop, '+repr(i-ai+1)+'/'+repr(bi-ai)+'  done...')

	print('cyclone.findcandidates:  candidates found, excluding less intense cyclones co-existing within the same sub-grid swlat / swlon  ...')
	# Check existence of many cyclones close to each other
	cindx=np.copy(cindx[1::,:].astype('int'))
	for t in range(0,int(tt)):
		if np.any(cindx[:,6]==t):
			# remove less intense cyclones co-existing within the same sub-grid swlat / swlon
			tcindx=np.copy(cindx[(cindx[:,6]==t),:])
			for k in range(0,tcindx.shape[0]):
				# 1) mean sea level pressure
				if candidates[0,t,tcindx[k,1],tcindx[k,4]] > 0 :
					[a,b]=np.where(candidates[0,t,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1]==1)
					for i in range(0,a.shape[0]):
						if prmslmsl[t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] > prmslmsl[t,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].min():
							candidates[0,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0
				# 2.1) 850mb absolute vorticity
				if candidates[1,t,tcindx[k,1],tcindx[k,4]] > 0 :
					[a,b]=np.where(candidates[1,t,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1]==1)
					for i in range(0,a.shape[0]):
						if lat[tcindx[k,1]]>0:
							if absvprs[t,0,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] < absvprs[t,0,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].max():
								candidates[1,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0
						else:
							if absvprs[t,0,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] > absvprs[t,0,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].min():
								candidates[1,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0
				# 2.2) 700mb absolute vorticity
				if candidates[2,t,tcindx[k,1],tcindx[k,4]] > 0 :
					[a,b]=np.where(candidates[2,t,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1]==1)
					for i in range(0,a.shape[0]):
						if lat[tcindx[k,1]]>0: # North Hemisphere
							if absvprs[t,1,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] < absvprs[t,1,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].max():
								candidates[2,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0
						else:  # South Hemisphere
							if absvprs[t,1,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] > absvprs[t,1,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].min():
								candidates[2,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0
				# 3.1) 850mb geopotential height
				if candidates[3,t,tcindx[k,1],tcindx[k,4]] > 0 :
					[a,b]=np.where(candidates[3,t,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1]==1)
					for i in range(0,a.shape[0]):
						if hgtprs[t,0,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] > hgtprs[t,0,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].min():
							candidates[3,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0
				# 3.2) 700mb geopotential height
				if candidates[4,t,tcindx[k,1],tcindx[k,4]] > 0 :
					[a,b]=np.where(candidates[4,t,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1]==1)
					for i in range(0,a.shape[0]):
						if hgtprs[t,1,tcindx[k,0]+a[i],tcindx[k,3]+b[i]] > hgtprs[t,1,tcindx[k,0]:tcindx[k,2]+1,tcindx[k,3]:tcindx[k,5]+1].min():
							candidates[4,t,tcindx[k,0]+a[i],tcindx[k,3]+b[i]]=0

	stop = timeit.default_timer()
	print('cyclone.findcandidates:  Concluded.  Duration: '+repr(int(round(stop - start,0)))+' seconds')
	return candidates


def position(*args):
	'''
 	2) Calculate cyclone positions (center, independent in time) and test cyclone candidates; strongly based on method of NCEP/NOAA:
	"HOW THE NCEP TROPICAL CYCLONE TRACKER WORKS", Timothy P. Marchok
	which uses pressure gradient and wind at 10m and at 850mb to restric/test cyclones previously identified.
	position is the second of a group of functions of cyclone tracking

	[candidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)

	10 inputs are requested:
	lat : vector containing latitudes, len(lat.shape) must be one.
	lon : vector containing longitudes, len(lon.shape) must be one.
	candidates : candidates matrix where 0 is non-cyclone and 1 is cyclone. NaN is excluded variable.
		candidates[0,:,:,:] : candidates found in the mean sea level pressure field
		candidates[1,:,:,:] : candidates found in the 850mb absolute vorticity
		candidates[2,:,:,:] : candidates found in the 700mb absolute vorticity
		candidates[3,:,:,:] : candidates found in the 850mb geopotential height
		candidates[4,:,:,:] : candidates found in the 700mb geopotential height

	prmslmsl : mean sea level pressure (one z level only). prmslmsl.shape  is  [time, latitudes and longitudes]
	ugrdprs : U-wind at 850mb. ugrdprs.shape can be [time, levels, latitudes and longitudes] or [time, latitudes and longitudes]
	vgrdprs : V-wind at 850mb. vgrdprs.shape can be [time, levels, latitudes and longitudes] or [time, latitudes and longitudes]
	ugrd10m : Surface U-wind at 10 meters. ugrdprs.shape is [time, latitudes and longitudes]
	vgrd10m : Surface V-wind at 10 meters. vgrdprs.shape is [time, latitudes and longitudes]
	restriction : vector of thresholds (minimum or maximum value) associated with criteria of restriction of each parameter (surface wind, pressure gradient, cyclonic wind at 850mb);
                   restriction[0] minimum intensity of maximum(space) wind inside the cyclone.
		   restriction[1] minimum pressure gradient (mb per km) present in all cyclonic directions. NCEP uses 1/333 or 1/100.
		   restriction[2] minimum averaged cyclonic wind at each of the 4 rays of the cyclone. The rays length is defined based on the movable sub-grid sw. NCEP uses 3 m/s.
		For Ex: restriction=[12.86,(1./333.),3.]
		The restriction and associated variable is turned off when entering zero; i.e., no restriction is applied.
	sw : vector containing the size (km) of the sub-grid that will run the entire grid/fields checking cyclone candidates.
		sw has size-in-latitude and size-in-longitude respectively
		For Ex: sw=np.array([275.,275.])

	User does not need to use all fields of prmslmsl, ugrdprs,vgrdprs and ugrd10m,vgrd10m. You can set as zero and the varible will be turned off (same effect of setting restriction as zero).

	2 outputs are provided;
	candidates : matrix with zero, one or two; with shape of variable/field, time, latitude, longitude.
		    0: zero means no cyclone found
                    1: one is the cyclone candidate position found.
                    2: cyclone confirmed and position calculated using centers from input candidates in each variable (from cyclone.findcandidates)
		    NaN: variable ignored
		    So a cyclone can be found in one parameter (prmslmsl for example) and not in the other (candidates=1) but the cyclone (candidates=2) must be identified and passed the restriction in all variables.
		candidates.shape[0] will always be 5, same as matrix from cyclone.findcandidates.
	cyclpos : array with time index and latitude and longitude of the cyclone identified

	Be careful in order to not be over-restrictive. Light criterias together can lead to very restrictive conditions and important cyclones can be missed.
	Cyclones are very different over the globe. Define certain thresholds for a family of cyclones within an especific region and different thresholds for other cases.

	Examples:

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 0, 85]; nt=57;
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		levels=[850,700]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdprs']
		[ugrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdprs']
		[vgrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','absvprs']
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs']
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1010.,5./100000.,2./100000.,1500.,3050.]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		restriction=[5.,(0.4/333.),0.] # restriction=[12.86,(1./333.),3]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)
 

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 10, 80]; nt=25;
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p50','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1030.,0,0,0,0]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,0,0,threshold,sw)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p50','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p50','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		restriction=[15,0,0]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,0,0,ugrd10m,vgrd10m,restriction,sw)
		forecastools.plotforecast(fcandidates[0,0,:,:],lat,lon) # show points of cyclone candidates from mean sea level pressure field at the first instant
                t=0                
                forecastools.plotforecast(fcandidates[0,t,:,:],lat,lon) # show points of final cyclone candidates from mean sea level pressure field at the instant t. You can play with that to visualize other times and variables. 
                forecastools.plotforecast(fcandidates[1,t,:,:],lat,lon) # absolute vorticity [1/s] at 850 mb
                forecastools.plotforecast(fcandidates[2,t,:,:],lat,lon) # absolute vorticity [1/s] at 700 mb
                forecastools.plotforecast(fcandidates[3,t,:,:],lat,lon) # geopotential height [geopotential meter, gpm] at 850 mb
                forecastools.plotforecast(fcandidates[4,t,:,:],lat,lon) # geopotential height [geopotential meter, gpm] at 700 mb

	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	if len(args) < 10:
		sys.exit(' ERROR! Insufficient input arguments. lat lon vectors; candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m matrices and restriction and sw vectors must be entered ')
	elif len(args) == 10:
		lat=copy.copy(args[0]); lon=copy.copy(args[1]); candidates=copy.copy(args[2]); prmslmsl=copy.copy(args[3]); ugrdprs=copy.copy(args[4]); vgrdprs=copy.copy(args[5]); ugrd10m=copy.copy(args[6]); vgrd10m=copy.copy(args[7]); restriction=copy.copy(args[8]); 
		if len(args[9])==2:
			swlat=copy.copy(args[9][0]); swlon=copy.copy(args[9][1])
		else:
			sys.exit(' ERROR! Problems with shape. Sw must have two values related to swlat and swlon. For ex: sw=np.array([275,275])')
	if len(lat.shape)>1 or len(lon.shape)>1:
		sys.exit(' ERROR! Problems with shape. Latitude and Longitude must be the first inputs')
	if len(candidates.shape)!=4:
		sys.exit(' ERROR! Wrong candidates matrix shape')

	if np.mean(restriction)==0:
		restriction=[0.,0.,0.]
		print(' No restriction will be applied')

	tt=[]
	if np.mean(prmslmsl)!=0:
		if len(prmslmsl.shape)!=3 :
			sys.exit(' ERROR! Problems with shape. It must be: len(prmslmsl.shape)=3 ')
		else:
			tt=np.append(tt,prmslmsl.shape[0])
	else:
		restriction[1]=0

	if np.mean(ugrd10m)!=0 and np.mean(vgrd10m)!=0:
		if len(ugrd10m.shape)<3 or len(vgrd10m.shape)<3:
			sys.exit(' ERROR! Problems with shape. It must be: len(ugrd10m.shape)=3, len(vgrd10m.shape)=3 ')
		else:
			tt=np.append(tt,ugrd10m.shape[0])
	else:
		restriction[0]=0

	if np.mean(ugrdprs)!=0 and np.mean(vgrdprs)!=0:
		if len(ugrdprs.shape)<3 or len(vgrdprs.shape)<3:
			sys.exit(' ERROR! Problems with shape. It must be: len(ugrdprs.shape)=3 or 4, len(vgrdprs.shape)=3 or 4,')
		else:
			tt=np.append(tt,ugrdprs.shape[0])
	else:
		restriction[2]=0

	if len(tt)!=0:
		if int(np.mean(tt)) == int(candidates.shape[1]):
			tt=int(np.mean(tt))
		else:
			sys.exit(' ERROR! Problems with shape. Your input fields/matrices does not have the same length in time as candidates')
	else:
		tt=candidates.shape[1]

	if len(restriction)!=3:
		sys.exit(' ERROR! Problems with shape. Restriction must have 3 inputs')

	# Only level=850mb is necessary
	if restriction[2] != 0:
		if len(ugrdprs.shape)==4:
			ugrdprs=np.copy(ugrdprs[:,0,:,:])
		if len(vgrdprs.shape)==4:
			vgrdprs=np.copy(vgrdprs[:,0,:,:])

	# Wind speed
	mgrd10m=np.sqrt(ugrd10m**2+vgrd10m**2)
	mgrdprs=np.sqrt(ugrdprs**2+vgrdprs**2)

	# Converting pressure to mb
	if np.mean(prmslmsl) > 95000:
		prmslmsl=np.copy(prmslmsl/100.)
	# searching sub-grid in meters
	swlat=swlat*1000.; swlon=swlon*1000.

	# From Cartesian to Spherical coordinates
	# distance in meters 
	latdist=np.zeros((lat.shape[0]),'f') # lat distance of grid points
	londist=np.zeros((lat.shape[0],lon.shape[0]),'f') # lon distance of grid points
	for i in range(0,lat.shape[0]):
		latdist[i]=forecastools.distance(lat[0],lon[0],lat[i],lon[0])
		for j in range(1,lon.shape[0]):
			londist[i,j]=londist[i,j-1]+forecastools.distance(lat[i],lon[j-1],lat[i],lon[j])
	londist[np.isnan(londist)==True]=0; latdist[np.isnan(latdist)==True]=0

	# variable with cyclone positions
	cyclpos=np.array([[0,0,0]],'f')
	# loop in time
	for t in range(0,int(tt)):
		# picking candidates only
		[a,b] = np.where(candidates[0,t,:,:]==1)	
		for c in range(0,len(a)):

			i=int(a[c]); j=int(b[c])
			posv=np.array([[i,j]],'i')
			# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
			aux=(latdist-latdist[i]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
			ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
			fi=int(np.where( abs((latdist-latdist[i])-swlat/2) == abs((latdist-latdist[i])-swlat/2).min() )[-1])  # final index lat
			aux=(londist[i,:]-londist[i,j]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
			ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
			fj=int(np.where( abs((londist[i,:]-londist[i,j])-swlon/2) == abs((londist[i,:]-londist[i,j])-swlon/2).min() )[-1])   # final index lon

			cp=1
			for k in range(1,candidates.shape[0]):
				if np.isnan(candidates[k,t,ii:fi+1,ij:fj+1]).min()==False :
					cp=cp+1
					if np.any(candidates[k,t,ii:fi+1,ij:fj+1]==1):
						[a1,b1] = np.where(candidates[k,t,ii:fi+1,ij:fj+1]==1)
						posv  = np.append(posv,np.array([[int(ii)+int(np.mean(a1)),int(ij)+int(np.mean(b1))]],'i'),axis=0)

			if posv.shape[0] == cp :
				indi = int(round(posv[:,0].mean())) ; indj = int(round(posv[:,1].mean()))
				candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=2
				# defining a running detection sub-grid (from user-defined distance in km) to identify and check cyclone candidates
				aux=(latdist-latdist[indi]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
				indii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
				indfi=int(np.where( abs((latdist-latdist[indi])-swlat/2) == abs((latdist-latdist[indi])-swlat/2).min() )[-1])  # final index lat
				aux=(londist[indi,:]-londist[indi,indj]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
				indij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
				indfj=int(np.where( abs((londist[indi,:]-londist[indi,indj])-swlon/2) == abs((londist[indi,:]-londist[indi,indj])-swlon/2).min() )[-1])   # final index lon

				# apply resctrictions criteria

				# 1) Surface wind within the cyclone is greater than restriction[0]
				if restriction[0] != 0. and mgrd10m[t,indii:indfi+1,ij:indfj+1].max()<restriction[0]:
					candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=0

				# 2) A minimum critical mslp gradient, extending in any direction from the center mslp position, must be found.
				#    For the AVN, in the NCEP tracker, this pressure gradient is 1 mb / 333 km.
				if restriction[1] != 0. :
					dd=abs(np.diff(latdist[indii:indi+1])/1000.) ; lg=np.diff(np.flipud(prmslmsl[t,indii:indi+1,j]))
					if sum(lg) < sum(restriction[1]*dd):
						candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1					

					dd=abs(np.diff(latdist[indi:indfi+1])/1000.) ; lg=np.diff(prmslmsl[t,indi:indfi+1,j])
					if sum(lg) < sum(restriction[1]*dd):
						candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1

					dd=abs(np.diff(londist[indi,indij:indj+1])/1000.) ; lg=np.diff(np.flipud(prmslmsl[t,indi,indij:indj+1]))
					if sum(lg) < sum(restriction[1]*dd):
						candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1

					dd=abs(np.diff(londist[indi,indj:indfj+1])/1000.) ; lg=np.diff(prmslmsl[t,indi,indj:indfj+1])
					if sum(lg) < sum(restriction[1]*dd):
						candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1

				# 3) For the 850 mb winds, the average tangential winds at 850 mb within sw km must be cyclonic with a minimum averaged intensity of restriction[2]
				#    According to NCEP, for AVN, at least 3 m/s
				if restriction[2] != 0. :
					if lat[i]>0:  # North Hemisphere,
						if ugrdprs[t,indii:indi+1,j].mean()<0. or mgrdprs[t,indii:indi+1,j].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
						if ugrdprs[t,indi:indfi+1,j].mean()>0. or mgrdprs[t,indi:indfi+1,j].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
						if vgrdprs[t,indi,indij:indj+1].mean()>0. or mgrdprs[t,indi,indij:indj+1].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
						if vgrdprs[t,indi,indj:indfj+1].mean()<0. or mgrdprs[t,indi,indj:indfj+1].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
					else:    # South Hemisphere,
						if ugrdprs[t,indii:indi+1,j].mean()>0. or mgrdprs[t,indii:indi+1,j].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
						if ugrdprs[t,indi:indfi+1,indj].mean()<0. or mgrdprs[t,indi:indfi+1,j].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
						if vgrdprs[t,indi,indij:indj+1].mean()<0. or mgrdprs[t,indi,indij:indj+1].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
						if vgrdprs[t,indi,indj:indfj+1].mean()>0. or mgrdprs[t,indi,indj:indfj+1].mean()<restriction[2]:
							candidates[(np.isnan(candidates[:,t,indi,indj])==False),t,indi,indj]=1
				# Write positions in the variable cyclpos 
				if candidates[0,t,indi,indj] == 2 :
					cyclpos = np.append(cyclpos,np.array([[t,lat[indi],lon[indj]]],'f'),axis=0)

	cyclpos = np.copy(cyclpos[1::,:])
	# remove possible repeated values
	auxc=abs(np.diff(cyclpos[:,0]))+abs(np.diff(cyclpos[:,1]))+abs(np.diff(cyclpos[:,2]))
	cyclpos = np.delete(cyclpos, find(auxc==0), axis=0)

	return candidates, cyclpos


def linktime(*args):
	'''
 	3) Take independent cyclone positions calculated with cyclone.position and link cyclones in time, creating tracks and cyclone evolution.

	fcyclones = cyclone.linktime(cyclpos,ntime,maxspeed,maxhdur,mindur)

	5 inputs are requested:
	cyclpos  : cyclone positions of independet cyclones calculated with cyclone.position. 
		   cyclopos is an array with [time ID, latitude, longitude] for each cyclone independent in time.
	ntime    : time array (in seconds), associated with previously used variables prmslmsl,absvprs,hgtprs,ugrdprs,vgrdprs,ugrd10m,vgrd10m, obtained with forecastools.getnoaaforecast
		   In case you do not have this time array, a single value containing the constant time interval in hours is allowed; for ex: ntime=3
	maxspeed : maximum propagation velocity (m/s) of the center of the cyclone, used to estipulate a limit distance to consider a cyclone at t+1 as a propagation of a cyclone at t. 
		   NCEP/NOAA uses 30.84 m/s (60knots), which is the value used in case maxspeed is set as zero.
	maxhdur  : maximum hidden duration (hours). It allows cyclone to not be detected at t+1 but still be consider a propagation of a cyclone t if it returns at t+2
		   Therefore a cyclone can be hidden during some time (or not detected for some reason) and still be linked to a cyclone that occurred before maxhdur
		   maxhdur=0 turns off this option so only cyclones occurring in sequence are allowed.
	mindur   : minimum cyclone duration (hours). It excludes all cyclones with duration less than mindur. 
		   mindur=0 ensures all cyclones are included.

	All five inputs a necessary. No parameter can be omitted. If you do not know just set as zero and this program will use default values. 

	1 output is provided;
	fcyclones : array with time index, latitude and longitude of the cyclone identified and the cyclone ID.
		    ntime[fcyclones[:,0]] are the exact time of the cyclone.

	Example:

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 0, 85]; nt=57;
		#aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime( time() - 24*60*60 ) ) # yesterday 
		initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		levels=[850,700]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdprs']
		[ugrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdprs']
		[vgrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','absvprs']
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs']
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1020.,5./100000.,2./100000.,1500.,3050.]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		restriction=[5,0.0001,0] # restriction=[12.86,(1./333.),3]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)
		maxspeed=20; maxhdur=6; mindur=24
		fcyclones = cyclone.linktime(cyclpos,ntime,maxspeed,maxhdur,mindur)

	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	if len(args) < 5:
		sys.exit(' ERROR! Insufficient input arguments. cyclpos, ntime, maxspeed, maxhdur and mindur be entered')
	else:
		cyclpos=copy.copy(args[0]); ntime=copy.copy(args[1]); maxspeed=copy.copy(args[2]); maxhdur=copy.copy(args[3]); mindur=copy.copy(args[4]);

	if len(cyclpos) <1:
		print(' Input Cyclopos is empty, no cyclone identified')
		fcyclpos = []
		return fcyclpos

	if len(ntime)==1:
		if ntime==0:
			ntime = np.linspace(1,cyclpos[:,0].max(),cyclpos[:,0].max())*3*3600
		else:
			ntime = np.linspace(1,cyclpos[:,0].max(),cyclpos[:,0].max())*ntime*3600

	if maxspeed==0:
		maxspeed = 60.*0.514;

	fcyclpos=np.zeros((cyclpos.shape[0],5),'f')
	fcyclpos[:,0:3]=cyclpos[:,:]

	cc=1
	a = np.where(cyclpos[:,0]==cyclpos[0,0])[0]
	for i in range(0,len(a)):
		fcyclpos[a[i],3]=cc
		cc=cc+1

	for t in range(0,ntime.shape[0]-1):

		dt=abs(ntime[t]-ntime[t+1]); dslim=maxspeed*dt
		a = np.where(cyclpos[:,0]==t)[0]
		b = np.where(cyclpos[:,0]==t+1)[0]
		sa=np.copy(a)*0 ; sb=np.copy(b)*0
		if np.any(cyclpos[:,0]==t+1):
			if np.any(cyclpos[:,0]==t):
				tm=np.zeros((len(a),len(b)),'f')
				for i in range(0,len(a)):
					for j in range(0,len(b)):
						tm[i,j]=forecastools.distance(cyclpos[a[i],1],cyclpos[a[i],2],cyclpos[b[j],1],cyclpos[b[j],2])
				k=np.where(tm<=dslim)
				if np.any(tm<=dslim):
					si = np.argsort(tm[k])
					for i in range(0,len(k[0])):
						# at t+1 the one cyclone is associated solely with one(nearest) cyclone at t, and vice-versa
						# if a cyclone at t+1 comes from 2 cyclones at t, it will be associated with the nearest one.
						# if a cyclone a t creates two cyclones at t+1, in t+1 only the nearest will be associated with that, the other will be consider a new independent cyclone.
						if sa[k[0][si[i]]] ==0 and sb[k[1][si[i]]] == 0 :
							fcyclpos[b[k[1][si[i]]],3] = fcyclpos[a[k[0][si[i]]],3]
							sa[k[0][si[i]]] = 1 ; sb[k[1][si[i]]] = 1

			for j in range(0,len(b)):
				if sb[j] == 0:
					if int((maxhdur*3600.)/abs(ntime[t]-ntime[t+1])) >= 2:
						for r in range(2,int((maxhdur*3600.)/abs(ntime[t]-ntime[t+1]))+1):
							if t>=r and abs(ntime[t+1]-ntime[t-r+1])<=maxhdur*3600. :
								dt=abs(ntime[t+1]-ntime[t-r+1]); dslim=maxspeed*dt
								na = np.where(cyclpos[:,0]==t-r+1)[0]; nsa=np.copy(na)*0;
								if np.any(cyclpos[:,0]==t-r+1):
									tm=np.zeros((len(na),len(b)),'f')
									for i in range(0,len(na)):
										for j in range(0,len(b)):
											tm[i,j]=forecastools.distance(cyclpos[na[i],1],cyclpos[na[i],2],cyclpos[b[j],1],cyclpos[b[j],2])
									k=np.where(tm<=dslim)
									if np.any(tm<=dslim):
										si = np.argsort(tm[k])
										for i in range(0,len(k[0])):
											if nsa[k[0][si[i]]] ==0 and sb[k[1][si[i]]] == 0 :
												fcyclpos[b[k[1][si[i]]],3] = fcyclpos[na[k[0][si[i]]],3]
												nsa[k[0][si[i]]] = 1 ; sb[k[1][si[i]]] = 1

			for j in range(0,len(b)):
				if sb[j] == 0 :
					fcyclpos[b[j],3] = cc
					cc = cc+1

		del b, a, sa, sb

	if mindur > 0:
		ch=1
		for i in range(0,int(fcyclpos[:,3].max())+1):
			c=np.where(fcyclpos[:,3].astype('i') == i)[0]
			if np.any(c):
				if (ntime[fcyclpos[c,0].astype('i')].max() - ntime[fcyclpos[c,0].astype('i')].min()) > mindur*3600. :
					fcyclpos[c,4]=ch
					ch=ch+1
	else:
		fcyclpos[:,4] = fcyclpos[:,3]

	fcyclpos = np.copy( fcyclpos[fcyclpos[:,4]>0][:,[0,1,2,4]] )
	return fcyclpos



def ctable(*args):
	'''
	4) Function to create a cyclone table with min, max and mean variable(data) within each cyclone. Developed to use results from cyclone.linktime.

	cyclone.ctable(fcyclones,sw,lat,lon,ntime,data)

	6 inputs are requested:
	fcyclones : array resulted from cyclone.linktime, containing position of cyclones. 
		    fcyclones[:,0] time ID
		    fcyclones[:,1] latitude of each cyclone at a specific instant.
		    fcyclones[:,2] longitude of each cyclone at a specific instant.
		    fcyclones[:,3] cyclone ID, linking the same cyclone at different time.

	sw : vector containing the size (km) of the sub-grid containing the cyclone.
		sw has size-in-latitude and size-in-longitude respectively
		For Ex: sw=np.array([275.,275.])

	lat : vector containing latitudes, len(lat.shape) must be one.
	lon : vector containing longitudes, len(lon.shape) must be one.

	ntime    : time array (in seconds), associated with previously used variables prmslmsl,absvprs,hgtprs,ugrdprs,vgrdprs,ugrd10m,vgrd10m, obtained with forecastools.getnoaaforecast

	data : matrix of any data to calculate mean, max and mean inside the cyclones. It can be mean sea level pressure, wind intensity, vorticity etc. For the plots it must be wind intensity.
	       data.shape must be equal to [ntime.shape, lat.shape, lon.shape]

	1 output is provided;
	ftable : array with the same length as fcyclones containing cycloneID, date, lat, lon and min,max and mean of input data within the cyclones.
	       ftable[:,0] cyclone ID
	       ftable[:,[1,2,3,4,5]] date [year,month,day,hour,minute]
	       ftable[:,6] latitude of the center of the cyclone
	       ftable[:,7] longitude of the center of the cyclone
	       ftable[:,8] minimum value of a given variable within the cyclone with size sw 
	       ftable[:,8] maximum value of a given variable within the cyclone with size sw 
	       ftable[:,8] mean value of a given variable within the cyclone with size sw 

	Example :

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 0, 85]; nt=57; 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		levels=[850,700]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdprs']
		[ugrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdprs']
		[vgrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','absvprs']
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs']
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1030.,5./100000.,2./100000.,1500.,3050.]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		restriction=[5,0.0001,0] # restriction=[12.86,(1./333.),3]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)
		maxspeed=30.*0.514; maxhdur=6; mindur=24
		fcyclones = cyclone.linktime(cyclpos,ntime,maxspeed,maxhdur,mindur)
		data = sqrt(ugrd10m**2 + vgrd10m**2)
		ftable = cyclone.ctable(fcyclones,sw,lat,lon,ntime,data)
		hd=' CycloneID, YEAR,MONTH,DAY,HOUR,MINUTE, Lat, Lon, minData, maxData, meanData'
		np.savetxt('cyclones.txt',ftable, fmt='%8.2f', delimiter='\t', newline='\n', header=hd, footer='', comments='# ')

	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	if len(args) < 6:
		sys.exit(' ERROR! Insufficient input arguments. fcyclones,sw,lat,lon,ntime and data must be provided')
	elif len(args) == 6:
		fcyclones=copy.copy(args[0]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); ntime=copy.copy(args[4]); data=copy.copy(args[5]);
		if len(args[1])==2:
			swlat=copy.copy(args[1][0]); swlon=copy.copy(args[1][1])
			swlat=swlat*1000.; swlon=swlon*1000.
		else:
			sys.exit(' ERROR! Problems with shape. Sw must have two values related to swlat and swlon. For ex: sw=np.array([275,275])')
	else:
		sys.exit(' ERROR! Too many arguments. fcyclones, lonlat and figinfo are necessary only')

	if lat.shape[0]!=data.shape[1] or lon.shape[0]!=data.shape[2] or ntime.shape[0]!=data.shape[0]:
		sys.exit(' ERROR! Problems with shape. These conditions must be satisfied: ntime.shape[0]==data.shape[0], lat.shape[0]==data.shape[1], lon.shape[0]==data.shape[2]')

	if len(fcyclones) == 0 :
		print(' ERROR! No cyclones in fcyclones, it is empty')

	# From Cartesian to Spherical coordinates , distance in meters 
	latdist=np.zeros((lat.shape[0]),'f') # lat distance of grid points
	londist=np.zeros((lat.shape[0],lon.shape[0]),'f') # lon distance of grid points
	for i in range(0,lat.shape[0]):
		latdist[i]=forecastools.distance(lat[0],lon[0],lat[i],lon[0])
		for j in range(1,lon.shape[0]):
			londist[i,j]=londist[i,j-1]+forecastools.distance(lat[i],lon[j-1],lat[i],lon[j])
	londist[np.isnan(londist)==True]=0; latdist[np.isnan(latdist)==True]=0

	# initial and final index defining the searching area
	ilati=find(latdist>swlat/2.)[0]
	ilatf=find( abs(latdist-latdist.max()) > swlat/2.)[-1]
	if np.diff(latdist[ilati:ilatf+1]).sum() < swlat :
		sys.exit(' ERROR! Area not large enough for the algorithm (latitude)')
	iloni=find(londist[np.where(abs(lat)==abs(lat).min())[0],:][0]>swlat/2.)[0]
	ilonf=find( abs(londist[np.where(abs(lat)==abs(lat).min())[0],:][0]-londist[np.where(abs(lat)==abs(lat).min())[0],:][0].max()) > swlon/2.)[-1]
	if np.diff(londist[np.where(abs(lat)==abs(lat).min())[0],iloni:ilonf+1]).sum() < swlon :
		sys.exit(' ERROR! Area not large enough for the algorithm (longitude)')

	j=0; ftable=np.zeros((fcyclones.shape[0],11),'f')
	# loop over cyclones
	for c in range(int(fcyclones[:,3].min()),int(fcyclones[:,3].max())+1):
		ci = np.where(fcyclones[:,3]==c)[0] # time index of each cyclone
		for i in range(0,len(ci)):
			# lat and lon indexes of specific cyclone positioning
			iclat= np.where( abs(lat-fcyclones[ci[i],1]) == abs(lat-fcyclones[ci[i],1]).min() )[0]
			iclon= np.where( abs(lon-fcyclones[ci[i],2]) == abs(lon-fcyclones[ci[i],2]).min() )[0]
			# check if lat lon of the cyclone is inside the data grid
			if np.any(iclat):
				if iclat>=ilati and iclat<=ilatf:
					if np.any(iclon):
						if iclon>=iloni and iclon<=ilonf:
							# cyclone ID , first column
							ftable[j,0]=fcyclones[ci[i],3].astype('i')
							# Date (year,month,day,hour,minute)
							ftable[j,1]=gmtime(ntime[fcyclones[ci[i],0].astype('i')])[0]
							ftable[j,2]=gmtime(ntime[fcyclones[ci[i],0].astype('i')])[1]
							ftable[j,3]=gmtime(ntime[fcyclones[ci[i],0].astype('i')])[2]
							ftable[j,4]=gmtime(ntime[fcyclones[ci[i],0].astype('i')])[3]
							ftable[j,5]=gmtime(ntime[fcyclones[ci[i],0].astype('i')])[4]
							ftable[j,6]=lat[iclat]
							ftable[j,7]=lon[iclon]
							# define grid to select min, max and mean
							aux=(latdist-latdist[iclat]); aux=np.copy(aux[aux<=0]); aux=aux*-1.;
							ii=int(np.where( abs(aux-(swlat/2)) == abs(aux-(swlat/2)).min() )[0])  # initial index lat
							fi=int(np.where( abs((latdist-latdist[iclat])-swlat/2) == abs((latdist-latdist[iclat])-swlat/2).min() )[-1])  # final index lat
							aux=(londist[iclat,:]-londist[iclat,iclon]); aux=np.copy(aux[aux<=0]); aux=aux*-1.
							ij=int(np.where( abs(aux-(swlon/2)) == abs(aux-(swlon/2)).min() )[0]); del aux   # initial index lon
							fj=int(np.where( abs((londist[iclat,:]-londist[iclat,iclon])-swlon/2) == abs((londist[iclat,:]-londist[iclat,iclon])-swlon/2).min() )[-1])   # final index lon

							daux=data[fcyclones[ci[i],0].astype('i'),ii:fi+1,ij:fj+1]
							ftable[j,8]=daux[(np.isnan(daux)==False)].min()
							ftable[j,9]=daux[(np.isnan(daux)==False)].max()
							ftable[j,10]=daux[(np.isnan(daux)==False)].mean()
							j=j+1
							del daux

			del iclat, iclon
	ftable = np.copy( ftable[ftable[:,0]>0] )

	return ftable


# Visualization Section. Cyclone Plots ******************************************************************
# *******************************************************************************************************
# Plots uses the Saffir Simpson hurricane wind scale:
# https://en.wikipedia.org/wiki/Saffir%E2%80%93Simpson_hurricane_wind_scale
# http://www.nhc.noaa.gov/aboutsshws.php
# http://hypertextbook.com/facts/StephanieStern.shtml


def cplot(*args):
	'''
	Function to plot cyclone tracks using python basemap (all cyclones in the same figure). Developed to use results from cyclone.ctable and a desirable background field can be entered.

	cyclone.cplot(ftable,bgfield,lat,lon,figinfo)
	cyclone.cplot(cycloneinfo,bgfield,lat,lon)

	4 or 5 inputs are requested:
	ftable : array resulted from cyclone.ctable, containing position of cyclones. When using ftable, all cyclones will be plotted.
		ftable[:,0] cyclone ID
		ftable[:,1],ftable[:,2],ftable[:,3],ftable[:,4],ftable[:,5] YEAR,MONTH,DAY,HOUR,MINUTE
		ftable[:,6] latitude
		ftable[:,7] longitude
		ftable[:,8] min value from cyclone data
		ftable[:,9] max value from cyclone data
		ftable[:,10] mean value from cyclone data

	cycloneinfo : instead of using ftable from cyclone.ctable containing information of all cyclones, a different array (cycloneinfo) can be used instead; containing information about only one cyclone.
		For Ex: idx=np.append(range(1,8),[9]); cycloneinfo = ftable[ftable[:,0]==1,:][:,idx] # for the first cyclone (1)
		cycloneinfo is organized with:
		cycloneinfo[:,0],cycloneinfo[:,1],cycloneinfo[:,2],cycloneinfo[:,3],cycloneinfo[:,4] YEAR,MONTH,DAY,HOUR,MINUTE
		cycloneinfo[:,5] latitude
		cycloneinfo[:,6] longitude
		cycloneinfo[:,7] maximum wind speed inside the cyclone

	bgfield : background field. A 2D matrix with one field to be plotted in background. If you do not have it, you can enter zero.
		For Ex: data=sqrt(ugrd10m**2 + vgrd10m**2); bgfield=np.copy(data.max(0))

	lon : longitude array of bgfield. In case you don't enter bgfield, you can create a lon array to define the map where cyclone(s) will be plotted. A zero value can also be used.
		For Ex: lon=np.linspace(70,110,100)
	lat : latitude array of bgfield. In case you don't enter bgfield, you can create a lat array to define the map where cyclone(s) will be plotted. A zero value can also be used.
		For Ex: lat=np.linspace(-20,10,100)

	figinfo : figure size for Basemap plot
		  For Ex: levw=linspace(0,bgfield.max(),21); figinfo = [(8,5),levw,0,0,'NOAA 10-m Wind (m/s)'] 
		  If you are using forecastools.plotforecast then you can enter the same figinfo.
		  figinfo can be omitted or set as zero. In this case, the default size will be used.

	Output consists of only one figure with cyclone plots using a color jet scale based on the wind intensity inside the cyclone.
		Use zoom-in and zoom-out to better visualize the track
		Figure can be saved using:
        	savefig('NameOfInterest.jpg', dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='jpg',transparent=False, bbox_inches=None, pad_inches=0.1)
		see other options of savefig

	Examples :

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 0, 85]; nt=57; 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		levels=[850,700]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdprs']
		[ugrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdprs']
		[vgrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','absvprs']
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs']
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1030.,5./100000.,2./100000.,1500.,3050.]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		restriction=[5,0.0001,0.001] # restriction=[12.86,(1./333.),3]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)
		maxspeed=30.*0.514; maxhdur=6; mindur=24
		fcyclones = cyclone.linktime(cyclpos,ntime,maxspeed,maxhdur,mindur)
		wspeed=sqrt(ugrd10m**2 + vgrd10m**2); bgfield=np.copy(wspeed.max(0))
		ftable = cyclone.ctable(fcyclones,sw,lat,lon,ntime,wspeed)
		levw=linspace(0,bgfield.max(),21); figinfo = [(8,5),levw,0,0,'NOAA 10-m Wind (m/s)'] 
		cyclone.cplot(ftable,bgfield,lat,lon,figinfo)
		cyclone.cplot(ftable,0,0,0)

	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	# parameters for the plot ---------------------
	paletteg = plt.cm.gray_r
	lldsp = 5.
	# wind intensity levels (for the cyclones). The last five are related to tropical cyclones (Saffir-Simpson Hurricane Wind Scale)
	clevels=[0,8,12,16,20,24,28,33,42,50,58,70]
	# parameters for the plot
	linwtype=['1','1','1','1','1','1','1','2','2','2','2','3']
	np.linspace(0,0.5,len(clevels)-5)
	sizemarko=np.append(np.linspace(0.3,0.6,len(clevels)-5), [0.68,0.76,0.84,0.92,1.0] )
	n=len(clevels)-4
	color=matplotlib.cm.jet(np.linspace(0,1,n))[0:-1,:]
	color = np.append(color,[[1.00, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.8, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.6, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.4, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.2, 0.0, 0.0, 1.0]],axis=0)
	# ---------------------------------------------

	if len(args) < 4:
		sys.exit(' ERROR! Insufficient input arguments. At least cycloneinfo, bgfield, lon and lat, must be provided. \
		In case you do not have bgfield, lon and lat, please enter 0 at each position')
	elif len(args) == 4:
		cycloneinfo=copy.copy(args[0]); bgfield=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=0
	elif len(args) == 5:
		cycloneinfo=copy.copy(args[0]); bgfield=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=copy.copy(args[4])
	else:
		sys.exit(' ERROR! Too many arguments. cycloneinfo, bgfield, lon, lat and figinfo are necessary only')

	if np.all(lon==0) or np.all(lat==0):
		lon=0; lat=0; bgfield=0

	if np.all(cycloneinfo==0) == True :
		print(' Warning! No cyclones in cycloneinfos, it is empty')
		cycloneinfo = 0
	elif np.atleast_2d(cycloneinfo).shape[1] != 8 and np.atleast_2d(cycloneinfo).shape[1] != 11:
		if np.atleast_2d(bgfield).shape[1] > 1:
			print(' Warning! Shape of cycloneinfo is wrong. It should be: YEAR, MONTH, DAY, HOUR, MIN, Lat, Lon, WindSpeed. \
			Or CycloneID YEAR, MONTH, DAY, HOUR, MIN, Lat, Lon, MinWind, MaxWind, MeanWind')
			cycloneinfo = 0
		else:
			sys.exit(' ERROR! No cyclones in cycloneinfos informed and no fields provided')
	# in case user enters directly the full table ftable from cyclone.ctable function
	if np.atleast_2d(cycloneinfo).shape[1] == 11:
		idx=np.append(range(0,8),[9])
		cycloneinfo = np.copy( cycloneinfo[:,idx])
	# in case user enters YEAR, MONTH, DAY, HOUR, MIN, Lat, Lon, WindSpeed
	elif np.atleast_2d(cycloneinfo).shape[1] == 8:
		cycloneinfo = np.copy( np.append(np.zeros([len(cycloneinfo),1])+1,cycloneinfo,1) )
	# check the existence of a background field to plot and if it is matching its lat and lon
	if len(np.atleast_2d(bgfield).shape) ==3:
		bgfield=np.copy(bgfield.mean(0))
	elif len(np.atleast_2d(bgfield).shape) ==4:
		bgfield=np.copy(bgfield.mean(0).mean(0))
	if np.atleast_2d(bgfield).shape[1] > 1:
		if np.atleast_2d(bgfield).shape[0] != np.atleast_1d(lat).shape[0] or np.atleast_2d(bgfield).shape[1] != np.atleast_1d(lon).shape[0]:
			print(' Warning! bgfield matrix size is not matching with lat and/or lon sizes. The background field will not be plotted')
			bgfield=0
		elif abs(np.array(bgfield)).max() > 9.99e+20:
			bgfield[abs(bgfield)>9.99e+20]=NaN
	# if lat and lon are entered with zero (len=1)
	if np.size(lat) == 1 or np.size(lon) == 1 :
		# in case the table with cyclones is provided
		if np.all(cycloneinfo==0) == False :
			# create a lat and lon array with limits of cyclone positions
			lat=np.linspace(cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].min()-lldsp,cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].max()+lldsp,101)
			lon=np.linspace(cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].min()-lldsp,cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].max()+lldsp,101)
			if lat.min() > (-90.+lldsp) and lat.max() < (90.-lldsp):			
				lat=np.linspace(cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].min()-lldsp,cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].max()+lldsp,101)
			if lon.min() > (-180.+lldsp) and lon.max() < (360.-lldsp):
				lon=np.linspace(cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].min()-lldsp,cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].max()+lldsp,101)
		else:
			sys.exit(' ERROR! No cyclones in cycloneinfo informed and wrong (or zero) lat/lon provided')

	# defining automatic figure parameters 
	# cyclone marker size 
	if (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) * (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())> (90**2):
		ms=50
	elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) * (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())> (45**2):
		ms=100
	elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) * (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())> (22**2):
		ms=150
	else:
		ms=200
	# figure info and levels (for the fields)
	lv=1
	if figinfo==0:
		if min(np.atleast_2d(bgfield).shape) > 1:
			if bgfield[np.isnan(bgfield)==False].mean() != 0.0:
				# level for cbar 
				levels = np.linspace( bgfield[(np.isnan(bgfield)==False)].min(), np.percentile(bgfield[(np.isnan(bgfield)==False)],99.99), 20)
				figinfo = [0,levels,0,0,'']
			else:
				figinfo = [0,0,0,0,'']; lv=0
		else:
			figinfo = [0,0,0,0,'']; lv=0
	elif len(np.atleast_1d(figinfo))==2 and min(np.atleast_2d(bgfield).shape) > 1:
		levels = np.linspace( bgfield[(np.isnan(bgfield)==False)].min(), np.percentile(bgfield[(np.isnan(bgfield)==False)],99.99), 20)
		figinfo = [figinfo,levels,0,0,'']
	elif np.atleast_1d(figinfo[1]).shape[0]==1 and min(np.atleast_2d(bgfield).shape) > 1:
		levels = np.linspace( bgfield[(np.isnan(bgfield)==False)].min(), np.percentile(bgfield[(np.isnan(bgfield)==False)],99.99), 20)
		figinfo[1] = levels

	# cbar string format and position
	if lv==1 and min(np.atleast_2d(bgfield).shape) > 1:
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
	if abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.7:
		db=0.08
	elif abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.5:
		db=0.06
	else:
		db=0.04


	# opening figure with figure size in case it is informed  
	if figinfo[0]==0:
		fig=plt.figure()		
	else:
		fig=plt.figure(figsize=figinfo[0])

	gs = gridspec.GridSpec(12,12)
	plt.subplot(gs[0:-2,0:-1])

	# PLOTS (3 projections depending on the lat/lon are possible) ------------------

	if (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())>350:
		# Global Plot with Eckert IV Projection	 ------------------

		if lon[np.isnan(lon)==False].max()>350 :
			bgfield[:,:],lon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),bgfield[:,:],lon,start=False)

		map = Basemap(projection='eck4',lon_0 = 0, resolution = 'l')
		[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
		xx, yy = map(mnlon,mnlat)
		# check if background field exists and is different than zero
		if min(np.atleast_2d(bgfield).shape) > 1:
			if bgfield[np.isnan(bgfield)==False].mean() != 0.0:
				map.bluemarble(scale=0.2); map.drawlsmask(ocean_color='w',land_color='None')
				map.contourf(xx,yy,bgfield,figinfo[1],cmap=paletteg,extend="max")
				ax = plt.gca()
				pos = ax.get_position()
				l, b, w, h = pos.bounds
				cax = plt.axes([l+0.07, b+0.04, w-0.15, 0.025]) # setup colorbar axes.
				cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat);# cbar.ax.tick_params(labelsize=fsmpb)
				tick_locator = ticker.MaxNLocator(nbins=7); cbar.locator = tick_locator; cbar.update_ticks()
				plt.axes(ax); cbar.set_label('Background field scale')
				map.contour(xx,yy,bgfield,cmap=paletteg)
			else:
				map.fillcontinents(color='grey')
		else:
			map.fillcontinents(color='grey')

		map.drawcoastlines(linewidth=0.8)
		map.drawcountries(linewidth=0.5)
		map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),40),labels=[0,0,0,1])
		map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),20),labels=[1,0,0,0])
		plt.title(figinfo[4])

		# check if cycloneinfo was provided or entered as zero. In case it is entered as zero, only the background field will be plotted
		if np.all(cycloneinfo==0) == False:
			# A loop through the cyclones. 
			for c in range(int(np.atleast_2d(cycloneinfo)[:,0].min()),int(np.atleast_2d(cycloneinfo)[:,0].max())+1):

				fcycloneinfo = np.copy( cycloneinfo[c==cycloneinfo[:,0],1::] )
				# check lat lon limits of cyclones and background fields
				if fcycloneinfo[:,5].min()<lat.min() or fcycloneinfo[:,5].max()>lat.max() or fcycloneinfo[:,6].min()<lon.min() or fcycloneinfo[:,6].max()>lon.max() :
					print(' Warning! Lat/Lon of cyclone out of the grid limits. You will not see in the plot')

				# marker size
				ms=30; sizemark=sizemarko*ms
				# continuous black line linking cyclones (o)
				x, y = map(fcycloneinfo[:,6],fcycloneinfo[:,5])
				map.plot(x,y,'k',linewidth=3,zorder=2)
				# added arrow 
				for i in range(0,fcycloneinfo.shape[0]-1):
					x,y = map(fcycloneinfo[i,6], fcycloneinfo[i,5])
					x2, y2 = map(fcycloneinfo[i+1,6],fcycloneinfo[i+1,5])
					plt.arrow(x,y,(x2-x)*0.5,(y2-y)*0.5,fc="k", ec="k",linewidth=3,zorder=2)
				# plot each cyclone with the proper colour scale associated with the intensity
				for i in range(1,fcycloneinfo.shape[0]-1):
					ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
					x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
					plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
					plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor=color[ind], linewidth=linwtype[ind], facecolor=color[ind],zorder=3)

				i=0 # first date (+)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='+', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatein = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatein, color='m', fontsize=7,verticalalignment='bottom',horizontalalignment='right',weight='bold',zorder=5)

				i=-1 # last date (x)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='x', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatefi, color='m', fontsize=7,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=5)
				# plot a small point at the center of the cyclone when entering a new day
				ind = np.where( np.diff(fcycloneinfo[:,2])!=0  )[0]+1
				if np.any(ind)==True:
					for t in range(0,ind.shape[0]):
						if ind[t]!=0 and ind[t]!=fcycloneinfo.shape[0]-1 :
							x, y = map(fcycloneinfo[ind[t],6],fcycloneinfo[ind[t],5])
							plt.scatter(x,y,marker='.', linewidth=1, zorder=4)
							aday=str(int(fcycloneinfo[ind[t],2])).zfill(2)
							# plt.text(x,y*0.98,aday, color='k',backgroundcolor='w', fontsize=4,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)
							# plt.text(x,y*(1.+(np.sign(y)*0.04)),aday, color='m',fontsize=7,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)

			# color bar of the cyclone wind scale clevels
			fm = np.zeros((lat.shape[0],lon.shape[0]),'f')-1
			map.contourf(xx,yy,fm,clevels,colors=color,extend="max")
			ax = plt.gca(); pos = ax.get_position()
			l, b, w, h = pos.bounds; cax = plt.axes([1.-l-(db/3.), b+0.05,0.02, h-0.1])
			cbar2=plt.colorbar(cax=cax, orientation='vertical',format='%4.0f',ticks=clevels)
			plt.axes(ax);  cbar2.set_label('Cyclonic Wind scale')


	elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())>20 and (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())<70 and abs(lat[np.isnan(lat)==False]).max()>60 and (float(lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())/float(lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()))<3. and (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())<60:
		# Regional High latitudes, Stereographic Projection ------------------

		# size/shape of the figure
		nwidth=int( (5000000*(lon[-1]-lon[0]))/50 )
		nheight=int( (5000000*(lat[-1]-lat[0]))/40 )
		# lat and lon parallels and meridians displacement
		lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
		latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )	

		# Basemap resolution
		fres='l'
		if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 50:
			fres='f'
		if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 100:
			fres='h'
		elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 500:
			fres='i'

		map = Basemap(width=nwidth, height=nheight, resolution=fres, projection='stere',
			lat_ts=lat[0], lat_0=((lat[-1]-lat[0])/2)+lat[0], lon_0=((lon[-1]-lon[0])/2)+lon[0]    )

		[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
		xx, yy = map(mnlon,mnlat)
		# check if background field exists and is different than zero
		if min(np.atleast_2d(bgfield).shape) > 1:
			if bgfield[np.isnan(bgfield)==False].mean() != 0.0:

				if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100:
					map.bluemarble(scale=np.tanh(4000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')

				map.contourf(xx,yy,bgfield,figinfo[1],cmap=paletteg,extend="max")

				ax = plt.gca(); pos = ax.get_position()
				l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
				cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
				tick_locator = ticker.MaxNLocator(nbins=7); cbar.locator = tick_locator; cbar.update_ticks()
				plt.axes(ax); cbar.set_label('Background field scale')

				map.contour(xx,yy,bgfield,cmap=paletteg)

			else:
				map.fillcontinents(color='grey')
		else:
			map.fillcontinents(color='grey')


		map.drawcoastlines(linewidth=0.8)
		map.drawcountries(linewidth=0.5)
		map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),lonmd),labels=[0,0,0,1])
		map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),latmd),labels=[1,0,0,0])
		plt.title(figinfo[4])

		# check if cycloneinfo was provided or entered as zero. In case it is entered as zero, only the background field will be plotted
		if np.all(cycloneinfo==0) == False:
			# A loop through the cyclones. 
			for c in range(int(np.atleast_2d(cycloneinfo)[:,0].min()),int(np.atleast_2d(cycloneinfo)[:,0].max())+1):

				fcycloneinfo = np.copy( cycloneinfo[c==cycloneinfo[:,0],1::] )
				# check lat lon limits of cyclones and background fields
				if fcycloneinfo[:,5].min()<lat.min() or fcycloneinfo[:,5].max()>lat.max() or fcycloneinfo[:,6].min()<lon.min() or fcycloneinfo[:,6].max()>lon.max() :
					print(' Warning! Lat/Lon of cyclone out of the grid limits. You will not see in the plot')

				# marker size
				sizemark=sizemarko*ms
				# continuous black line linking cyclones (o)
				x, y = map(fcycloneinfo[:,6],fcycloneinfo[:,5])
				map.plot(x,y,'k',linewidth=3,zorder=2)
				# added arrow 
				for i in range(0,fcycloneinfo.shape[0]-1):
					x,y = map(fcycloneinfo[i,6], fcycloneinfo[i,5])
					x2, y2 = map(fcycloneinfo[i+1,6],fcycloneinfo[i+1,5])
					plt.arrow(x,y,(x2-x)*0.5,(y2-y)*0.5,fc="k", ec="k",linewidth=3,zorder=2)
				# plot each cyclone with the proper colour scale associated with the intensity
				for i in range(1,fcycloneinfo.shape[0]-1):
					ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
					x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
					plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
					plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor=color[ind], linewidth=linwtype[ind], facecolor=color[ind],zorder=3)

				i=0 # first date (+)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='+', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatein = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatein, color='m', fontsize=7,verticalalignment='bottom',horizontalalignment='right',weight='bold',zorder=5)

				i=-1 # last date (x)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='x', facecolor='black', linewidth=1, zorder=4)	
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatefi, color='m', fontsize=7,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=5)
				# plot a small point at the center of the cyclone when entering a new day
				ind = np.where( np.diff(fcycloneinfo[:,2])!=0  )[0]+1
				if np.any(ind)==True:
					for t in range(0,ind.shape[0]):
						if ind[t]!=0 and ind[t]!=fcycloneinfo.shape[0]-1 :
							x, y = map(fcycloneinfo[ind[t],6],fcycloneinfo[ind[t],5])
							plt.scatter(x,y,marker='.', linewidth=1, zorder=4)
							aday=str(int(fcycloneinfo[ind[t],2])).zfill(2)
							# plt.text(x,y*0.98,aday, color='k',backgroundcolor='w', fontsize=7,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)
							plt.text(x,y*(1.+(np.sign(y)*0.04)),aday, color='m',fontsize=8,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)

			# color bar of the cyclone wind scale clevels
			fm = np.zeros((lat.shape[0],lon.shape[0]),'f')-1
			map.contourf(xx,yy,fm,clevels,colors=color,extend="max")
			ax = plt.gca(); pos = ax.get_position()
			l, b, w, h = pos.bounds; cax = plt.axes([1.-l-(db/3.), b+0.05,0.02, h-0.1])
			cbar2=plt.colorbar(cax=cax, orientation='vertical',format='%4.0f',ticks=clevels)
			plt.axes(ax);  cbar2.set_label('Cyclonic Wind scale')


	else:
		# Regional, Equidistant Cylindrical Projection ------------------

		# lat and lon parallels and meridians displacement
		lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
		latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )
		# Basemap resolution
		fres='l'
		if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 300:
			fres='i'
		elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 10:
			fres='h'

		map = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution=fres)

		[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
		xx, yy = map(mnlon,mnlat)
		# check if background field exists and is different than zero
		if min(np.atleast_2d(bgfield).shape) > 1:
			if bgfield[np.isnan(bgfield)==False].mean() != 0.0:

				if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100 and lon[-1]<=180:
					map.bluemarble(scale=np.tanh(5000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')

				map.contourf(xx,yy,bgfield,figinfo[1],cmap=paletteg,extend="max")

				ax = plt.gca(); pos = ax.get_position()
				l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
				cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
				tick_locator = ticker.MaxNLocator(nbins=7); cbar.locator = tick_locator; cbar.update_ticks()
				plt.axes(ax); cbar.set_label('Background field scale')

				map.contour(xx,yy,bgfield,cmap=paletteg)

			else:
				map.fillcontinents(color='grey')
		else:
			map.fillcontinents(color='grey')

		map.drawcoastlines(linewidth=0.8)
		map.drawcountries(linewidth=0.5)
		map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),lonmd),labels=[0,0,0,1])
		map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),latmd),labels=[1,0,0,0])
		plt.title(figinfo[4])


		# check if cycloneinfo was provided or entered as zero. In case it is entered as zero, only the background field will be plotted
		if np.all(cycloneinfo==0) == False:
			# A loop through the cyclones. 
			for c in range(int(np.atleast_2d(cycloneinfo)[:,0].min()),int(np.atleast_2d(cycloneinfo)[:,0].max())+1):

				fcycloneinfo = np.copy( cycloneinfo[c==cycloneinfo[:,0],1::] )
				# check lat lon limits of cyclones and background fields
				if fcycloneinfo[:,5].min()<lat.min() or fcycloneinfo[:,5].max()>lat.max() or fcycloneinfo[:,6].min()<lon.min() or fcycloneinfo[:,6].max()>lon.max() :
					print(' Warning! Lat/Lon of cyclone out of the grid limits. You will not see in the plot')


				# marker size
				sizemark=sizemarko*ms
				# continuous black line linking cyclones (o)
				x, y = map(fcycloneinfo[:,6],fcycloneinfo[:,5])
				map.plot(x,y,'k',linewidth=3,zorder=2)
				# added arrow 
				for i in range(0,fcycloneinfo.shape[0]-1):
					x,y = map(fcycloneinfo[i,6], fcycloneinfo[i,5])
					x2, y2 = map(fcycloneinfo[i+1,6],fcycloneinfo[i+1,5])
					plt.arrow(x,y,(x2-x)*0.5,(y2-y)*0.5,fc="k", ec="k",linewidth=3,zorder=2)
				# plot each cyclone with the proper colour scale associated with the intensity
				for i in range(1,fcycloneinfo.shape[0]-1):
					ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
					x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
					plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
					plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor=color[ind], linewidth=linwtype[ind], facecolor=color[ind],zorder=3)

				i=0 # first date (+)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='+', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatein = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatein, color='m', fontsize=7,verticalalignment='bottom',horizontalalignment='right',weight='bold',zorder=6)
				# plt.text(x,y, cdatein, color='k', fontsize=12,verticalalignment='bottom',horizontalalignment='right',weight='medium',zorder=6)

				i=-1 # last date (x)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='x', facecolor='black', linewidth=1, zorder=4)	
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatefi, color='m', fontsize=7,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)
				# plt.text(x,y, cdatefi, color='k', fontsize=12,verticalalignment='bottom',horizontalalignment='left',weight='medium',zorder=6)
				# plot a small point at the center of the cyclone when entering a new day
				ind = np.where( np.diff(fcycloneinfo[:,2])!=0  )[0]+1
				if np.any(ind)==True:
					for t in range(0,ind.shape[0]):
						if ind[t]!=0 and ind[t]!=fcycloneinfo.shape[0]-1 :
							x, y = map(fcycloneinfo[ind[t],6],fcycloneinfo[ind[t],5])
							plt.scatter(x,y,marker='.', linewidth=1, zorder=4)
							aday=str(int(fcycloneinfo[ind[t],2])).zfill(2)
							#plt.text(x,y*0.98,aday, color='k',backgroundcolor='w', fontsize=7,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)
							plt.text(x,y*(1.+(np.sign(y)*0.04)),aday, color='m',fontsize=8,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)

			# color bar of the cyclone wind scale clevels
			fm = np.zeros((lat.shape[0],lon.shape[0]),'f')-1
			map.contourf(xx,yy,fm,clevels,colors=color,extend="max")
			ax = plt.gca(); pos = ax.get_position()
			l, b, w, h = pos.bounds; cax = plt.axes([1.-l-(db/3.), b+0.05,0.02, h-0.1])
			cbar2=plt.colorbar(cax=cax, orientation='vertical',format='%4.0f',ticks=clevels)
			plt.axes(ax);  cbar2.set_label('Cyclonic Wind scale')


		plt.show()

	return map


def cplotic(*args):
	'''
	Function to plot cyclone tracks using python basemap (one figure per cyclone). Developed to use results from cyclone.ctable and a desirable background field can be entered.
	"cplot individual cyclones"

	cyclone.cplot(ftable,bgfield,lat,lon,figinfo)
	cyclone.cplot(cycloneinfo,bgfield,lat,lon)

	4 or 5 inputs are requested:
	ftable : array resulted from cyclone.ctable, containing position of cyclones. When using ftable, all cyclones will be plotted.
		ftable[:,0] cyclone ID
		ftable[:,1],ftable[:,2],ftable[:,3],ftable[:,4],ftable[:,5] YEAR,MONTH,DAY,HOUR,MINUTE
		ftable[:,6] latitude
		ftable[:,7] longitude
		ftable[:,8] min value from cyclone data
		ftable[:,9] max value from cyclone data
		ftable[:,10] mean value from cyclone data

	cycloneinfo : instead of using ftable from cyclone.ctable containing information of all cyclones, a different array (cycloneinfo) can be used instead; containing information about only one cyclone.
		For Ex: idx=np.append(range(1,8),[9]); cycloneinfo = ftable[ftable[:,0]==1,:][:,idx] # for the first cyclone (1)
		cycloneinfo is organized with:
		cycloneinfo[:,0],cycloneinfo[:,1],cycloneinfo[:,2],cycloneinfo[:,3],cycloneinfo[:,4] YEAR,MONTH,DAY,HOUR,MINUTE
		cycloneinfo[:,5] latitude
		cycloneinfo[:,6] longitude
		cycloneinfo[:,7] maximum wind speed inside the cyclone

	bgfield : background field. A 2D matrix with one field to be plotted in background. If you do not have it, you can enter zero.
		For Ex: data=sqrt(ugrd10m**2 + vgrd10m**2); bgfield=np.copy(data.max(0))

	lon : longitude array of bgfield. In case you don't enter bgfield, you can create a lon array to define the map where cyclone(s) will be plotted. A zero value can also be used.
		For Ex: lon=np.linspace(70,110,100)
	lat : latitude array of bgfield. In case you don't enter bgfield, you can create a lat array to define the map where cyclone(s) will be plotted. A zero value can also be used.
		For Ex: lat=np.linspace(-20,10,100)

	figinfo : figure size for Basemap plot
		  For Ex: levw=linspace(0,bgfield.max(),21); figinfo = [(8,5),levw,0,0,'NOAA 10-m Wind (m/s)'] 
		  If you are using forecastools.plotforecast then you can enter the same figinfo.
		  figinfo can be omitted or set as zero. In this case, the default size will be used.

	Output consists of one figure per cyclone plots using a color jet scale based on the wind intensity inside the cyclone.
		Use zoom-in and zoom-out to better visualize the track
		Figure can be saved using:
        	savefig('NameOfInterest.jpg', dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='jpg',transparent=False, bbox_inches=None, pad_inches=0.1)
		see other options of savefig

	Examples :

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 0, 85]; nt=57; 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		levels=[850,700]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdprs']
		[ugrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdprs']
		[vgrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','absvprs']
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs']
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1030.,5./100000.,2./100000.,1500.,3050.]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		restriction=[5,0.0001,0.001] # restriction=[12.86,(1./333.),3]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)
		maxspeed=30.*0.514; maxhdur=6; mindur=24
		fcyclones = cyclone.linktime(cyclpos,ntime,maxspeed,maxhdur,mindur)
		wspeed=sqrt(ugrd10m**2 + vgrd10m**2); bgfield=np.copy(wspeed.max(0))
		ftable = cyclone.ctable(fcyclones,sw,lat,lon,ntime,wspeed)
		levw=linspace(0,bgfield.max(),21); figinfo = [(8,5),levw,0,0,'NOAA 10-m Wind (m/s)'] 
		cyclone.cplotic(ftable,bgfield,lat,lon,figinfo)
		cyclone.cplotic(ftable,0,0,0)

	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	# parameters for the plot ---------------------
	paletteg = plt.cm.gray_r
	lldsp = 5.
	# wind intensity levels (for the cyclones). The last five are related to tropical cyclones (Saffir-Simpson Hurricane Wind Scale)
	clevels=[0,8,12,16,20,24,28,33,42,50,58,70]
	# parameters for the plot
	linwtype=['1','1','1','1','1','1','1','2','2','2','2','3']
	np.linspace(0,0.5,len(clevels)-5)
	sizemarko=np.append(np.linspace(0.3,0.6,len(clevels)-5), [0.68,0.76,0.84,0.92,1.0] )
	n=len(clevels)-4
	color=matplotlib.cm.jet(np.linspace(0,1,n))[0:-1,:]
	color = np.append(color,[[1.00, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.8, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.6, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.4, 0.0, 0.0, 1.0]],axis=0)
	color = np.append(color,[[0.2, 0.0, 0.0, 1.0]],axis=0)
	# ---------------------------------------------


	if len(args) < 4:
		sys.exit(' ERROR! Insufficient input arguments. At least cycloneinfo, bgfield, lon and lat, must be provided. \
		In case you do not have bgfield, lon and lat, please enter 0 at each position')
	elif len(args) == 4:
		cycloneinfo=copy.copy(args[0]); bgfield=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=0
	elif len(args) == 5:
		cycloneinfo=copy.copy(args[0]); bgfield=copy.copy(args[1]); lat=copy.copy(args[2]); lon=copy.copy(args[3]); figinfo=copy.copy(args[4])
	else:
		sys.exit(' ERROR! Too many arguments. cycloneinfo, bgfield, lon, lat and figinfo are necessary only')

	if np.all(lon==0) or np.all(lat==0):
		lon=0; lat=0; bgfield=0

	if np.all(cycloneinfo==0) == True :
		print(' Warning! No cyclones in cycloneinfos, it is empty')
		cycloneinfo = 0
	elif np.atleast_2d(cycloneinfo).shape[1] != 8 and np.atleast_2d(cycloneinfo).shape[1] != 11:
		if np.atleast_2d(bgfield).shape[1] > 1:
			print(' Warning! Shape of cycloneinfo is wrong. It should be: YEAR, MONTH, DAY, HOUR, MIN, Lat, Lon, WindSpeed. \
			Or CycloneID YEAR, MONTH, DAY, HOUR, MIN, Lat, Lon, MinWind, MaxWind, MeanWind')
			cycloneinfo = 0
		else:
			sys.exit(' ERROR! No cyclones in cycloneinfos informed and no fields provided')
	# in case user enters directly the full table ftable from cyclone.ctable function
	if np.atleast_2d(cycloneinfo).shape[1] == 11:
		idx=np.append(range(0,8),[9])
		cycloneinfo = np.copy( cycloneinfo[:,idx])
	# in case user enters YEAR, MONTH, DAY, HOUR, MIN, Lat, Lon, WindSpeed
	elif np.atleast_2d(cycloneinfo).shape[1] == 8:
		cycloneinfo = np.copy( np.append(np.zeros([len(cycloneinfo),1])+1,cycloneinfo,1) )
	# check the existence of a background field to plot and if it is matching its lat and lon
	if len(np.atleast_2d(bgfield).shape) ==3:
		bgfield=np.copy(bgfield.mean(0))
	elif len(np.atleast_2d(bgfield).shape) ==4:
		bgfield=np.copy(bgfield.mean(0).mean(0))
	if np.atleast_2d(bgfield).shape[1] > 1:
		if np.atleast_2d(bgfield).shape[0] != np.atleast_1d(lat).shape[0] or np.atleast_2d(bgfield).shape[1] != np.atleast_1d(lon).shape[0]:
			print(' Warning! bgfield matrix size is not matching with lat and/or lon sizes. The background field will not be plotted')
			bgfield=0
	# if lat and lon are entered with zero (len=1)
	if np.size(lat) == 1 or np.size(lon) == 1 :
		# in case the table with cyclones is provided
		if np.all(cycloneinfo==0) == False :
			# create a lat and lon array with limits of cyclone positions
			lat=np.linspace(cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].min()-lldsp,cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].max()+lldsp,101)
			lon=np.linspace(cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].min()-lldsp,cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].max()+lldsp,101)
			if lat.min() > (-90.+lldsp) and lat.max() < (90.-lldsp):			
				lat=np.linspace(cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].min()-lldsp,cycloneinfo[:,6][(np.isnan(cycloneinfo[:,6])==False) & (cycloneinfo[:,6]>-9999999)].max()+lldsp,101)
			if lon.min() > (-180.+lldsp) and lon.max() < (360.-lldsp):
				lon=np.linspace(cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].min()-lldsp,cycloneinfo[:,7][(np.isnan(cycloneinfo[:,7])==False) & (cycloneinfo[:,7]>-9999999)].max()+lldsp,101)
		else:
			sys.exit(' ERROR! No cyclones in cycloneinfo informed and wrong (or zero) lat/lon provided')


	# figure info and levels (for the fields)
	lv=1
	if figinfo==0:
		if min(np.atleast_2d(bgfield).shape) > 1:
			if bgfield[np.isnan(bgfield)==False].mean() != 0.0:
				# level for cbar 
				levels = np.linspace( bgfield[(np.isnan(bgfield)==False)].min(), np.percentile(bgfield[(np.isnan(bgfield)==False)],99.99), 20)
				figinfo = [0,levels,0,0,'']
			else:
				figinfo = [0,0,0,0,'']; lv=0
		else:
			figinfo = [0,0,0,0,'']; lv=0
	elif len(np.atleast_1d(figinfo))==2 and min(np.atleast_2d(bgfield).shape) > 1:
		levels = np.linspace( bgfield[(np.isnan(bgfield)==False)].min(), np.percentile(bgfield[(np.isnan(bgfield)==False)],99.99), 20)
		figinfo = [figinfo,levels,0,0,'']
	elif np.atleast_1d(figinfo[1]).shape[0]==1 and min(np.atleast_2d(bgfield).shape) > 1:
		levels = np.linspace( bgfield[(np.isnan(bgfield)==False)].min(), np.percentile(bgfield[(np.isnan(bgfield)==False)],99.99), 20)
		figinfo[1] = levels


	# cbar string format and position
	if lv==1 and min(np.atleast_2d(bgfield).shape) > 1:
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


	amap=[]; # array with each Basemap map from each figure, which is concatenated in a final array amap
	lato=np.copy(lat);lono=np.copy(lon)
	# A loop through the cyclones. One figure per cyclone.
	for c in range(int(np.atleast_2d(cycloneinfo)[:,0].min()),int(np.atleast_2d(cycloneinfo)[:,0].max())+1):

		# check if cycloneinfo was provided or entered as zero. In case it is entered as zero, only the background field will be plotted
		if np.all(cycloneinfo==0) == False:
			fcycloneinfo = np.copy( cycloneinfo[c==cycloneinfo[:,0],1::] )
			# check lat lon limits of cyclones and background fields
			if fcycloneinfo[:,5].min()<lato.min() or fcycloneinfo[:,5].max()>lato.max() or fcycloneinfo[:,6].min()<lono.min() or fcycloneinfo[:,6].max()>lono.max() :
				print(' Warning! Lat/Lon of cyclone out of the grid limits. You will not see in the plot')

			# define limits of the domain
			ilat=fcycloneinfo[:,5][(np.isnan(fcycloneinfo[:,5])==False) & (fcycloneinfo[:,5]>-9999999)].min()
			flat=fcycloneinfo[:,5][(np.isnan(fcycloneinfo[:,5])==False) & (fcycloneinfo[:,5]>-9999999)].max()
			ilon=fcycloneinfo[:,6][(np.isnan(fcycloneinfo[:,6])==False) & (fcycloneinfo[:,6]>-9999999)].min()
			flon=fcycloneinfo[:,6][(np.isnan(fcycloneinfo[:,6])==False) & (fcycloneinfo[:,6]>-9999999)].max()
			if (flat-ilat)<3. or (flon-ilon)<3.:
				lldsp=lldsp+3.

			sll = (flat-ilat)-(flon-ilon)
			ilat=ilat-lldsp; flat=flat+lldsp
			ilon=ilon-lldsp-sll/2.-1.; flon=flon+lldsp+sll/2.+1.	

			indilon=int(min(np.where( np.abs(lono-ilon)==np.min(np.abs(lono-ilon)))[0]))
			indflon=int(max(np.where( np.abs(lono-flon)==np.min(np.abs(lono-flon)))[0]+1))
			indilat=int(min(np.where( np.abs(lato-ilat)==np.min(np.abs(lato-ilat)))[0]))
			indflat=int(max(np.where( np.abs(lato-flat)==np.min(np.abs(lato-flat)))[0]+1))

			lat=np.copy(lato[indilat:indflat]); lon=np.copy(lono[indilon:indflon])
			if np.min(np.atleast_2d(bgfield).shape) > 1:
				bgfieldp=np.copy(bgfield[indilat:indflat,indilon:indflon])

		else:
			fcycloneinfo = 0
			lat=np.copy(lato); lon=np.copy(lono)

		# defining automatic figure parameters 
		# cyclone marker size 
		if (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) * (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())> (90**2):
			ms=50
		elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) * (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())> (45**2):
			ms=100
		elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) * (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())> (22**2):
			ms=150
		else:
			ms=200

		if abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.7:
			db=0.08
		elif abs(lat[-1]-lat[0])>abs(lon[-1]-lon[0])*0.5:
			db=0.06
		else:
			db=0.04

		if figinfo[0]==0:
			fig=plt.figure()		
		else:
			fig=plt.figure(figsize=figinfo[0])

		gs = gridspec.GridSpec(12,12)
		plt.subplot(gs[0:-2,0:-1])

		# PLOTS (3 projections depending on the lat/lon are possible) ------------------
		if (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())>350:

			# Global Plot with Eckert IV Projection	 ------------------

			if lon[np.isnan(lon)==False].max()>350 :
				bgfieldp[:,:],lon = shiftgrid(180.+(np.diff(lon[np.isnan(lon)==False]).mean()/2.),bgfieldp[:,:],lon,start=False)

			map = Basemap(projection='eck4',lon_0 = 0, resolution = 'l')
			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			# check if background field exists and is different than zero
			if min(np.atleast_2d(bgfield).shape) > 1:
				if bgfieldp[np.isnan(bgfieldp)==False].mean() != 0.0:

					map.bluemarble(scale=0.2); map.drawlsmask(ocean_color='w',land_color='None')

					map.contourf(xx,yy,bgfieldp,figinfo[1],cmap=paletteg,extend="max")

					ax = plt.gca()
					pos = ax.get_position()
					l, b, w, h = pos.bounds
					cax = plt.axes([l+0.07, b+0.04, w-0.15, 0.025]) # setup colorbar axes.
					cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat);# cbar.ax.tick_params(labelsize=fsmpb)
					tick_locator = ticker.MaxNLocator(nbins=7); cbar.locator = tick_locator; cbar.update_ticks()
					plt.axes(ax); cbar.set_label('Background field scale')

					map.contour(xx,yy,bgfieldp,cmap=paletteg)

				else:
					map.fillcontinents(color='grey')
			else:
				map.fillcontinents(color='grey')

			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),40),labels=[0,0,0,1])
			map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),20),labels=[1,0,0,0])

			# check if cycloneinfo was entered
			if np.all(fcycloneinfo==0) == False:
				# marker size
				ms=30; sizemark=sizemarko*ms
				# continuous black line linking cyclones (o)
				x, y = map(fcycloneinfo[:,6],fcycloneinfo[:,5])
				map.plot(x,y,'k',linewidth=3,zorder=2)
				# plot each cyclone with the proper colour scale associated with the intensity
				for i in range(1,fcycloneinfo.shape[0]-1):
					ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
					x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
					plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
					# plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=3)
					plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor=color[ind], linewidth=linwtype[ind], facecolor=color[ind],zorder=3)


				i=0 # first date (+)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='+', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatein = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				# plt.text(x,y, cdate, color='k', fontsize=12,horizontalalignment='left',weight='bold',zorder=4)

				i=-1 # last date (x)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='x', facecolor='black', linewidth=1, zorder=4)	
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				# plt.text(x,y, cdate, color='k', fontsize=12,horizontalalignment='left',weight='bold',zorder=4)
				# plot a small point at the center of the cyclone when entering a new day
				ind = np.where( np.diff(fcycloneinfo[:,2])!=0  )[0]+1
				if np.any(ind)==True:
					for t in range(0,ind.shape[0]):
						x, y = map(fcycloneinfo[ind[t],6],fcycloneinfo[ind[t],5])
						plt.scatter(x,y,marker='.', edgecolor='black', linewidth=1, zorder=4)
						#aday=str(int(fcycloneinfo[ind[t],2])).zfill(2)
						#plt.text(x,y*(1.+(np.sign(y)*0.04)),aday, color='m',fontsize=9,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)

				# color bar of the cyclone wind scale clevels
				fm = np.zeros((lat.shape[0],lon.shape[0]),'f')-1
				map.contourf(xx,yy,fm,clevels,colors=color,extend="max")
				ax = plt.gca(); pos = ax.get_position()
				l, b, w, h = pos.bounds; cax = plt.axes([1.-l-(db/3.), b+0.05,0.02, h-0.1])
				cbar2=plt.colorbar(cax=cax, orientation='vertical',format='%4.0f',ticks=clevels)
				plt.axes(ax);  cbar2.set_label('Cyclonic Wind scale')

				plt.title(figinfo[4]+'  Cyclone '+cdatein+' (+) to '+cdatefi+' (x)')

			else:
				plt.title(figinfo[4])

			amap = np.append(amap,map); # concatenate Basemaps of each figure (each cyclonpe)

		elif (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())>20 and (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min())<70 and abs(lat[np.isnan(lat)==False]).max()>60 and (float(lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())/float(lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()))<3. and (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min())<60:
			# Regional High latitudes, Stereographic Projection ------------------

			# size/shape of the figure
			nwidth=int( (5000000*(lon[-1]-lon[0]))/50 )
			nheight=int( (5000000*(lat[-1]-lat[0]))/40 )
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )	

			# Basemap resolution
			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 50:
				fres='f'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 100:
				fres='h'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 500:
				fres='i'

			map = Basemap(width=nwidth, height=nheight, resolution=fres, projection='stere',
				lat_ts=lat[0], lat_0=((lat[-1]-lat[0])/2)+lat[0], lon_0=((lon[-1]-lon[0])/2)+lon[0]    )

			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			# check if background field exists and is different than zero
			if min(np.atleast_2d(bgfield).shape) > 1:
				if bgfieldp[np.isnan(bgfieldp)==False].mean() != 0.0:

					if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100:
						map.bluemarble(scale=np.tanh(4000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')

					map.contourf(xx,yy,bgfieldp,figinfo[1],cmap=paletteg,extend="max")

					ax = plt.gca(); pos = ax.get_position()
					l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
					cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
					tick_locator = ticker.MaxNLocator(nbins=7); cbar.locator = tick_locator; cbar.update_ticks()
					plt.axes(ax); cbar.set_label('Background field scale')

					map.contour(xx,yy,bgfieldp,cmap=paletteg)

				else:
					map.fillcontinents(color='grey')
			else:
				map.fillcontinents(color='grey')


			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),lonmd),labels=[0,0,0,1])
			map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),latmd),labels=[1,0,0,0])

			# check if cycloneinfo was entered
			if np.all(fcycloneinfo==0) == False:
				# marker size
				sizemark=sizemarko*ms
				# continuous black line linking cyclones (o)
				x, y = map(fcycloneinfo[:,6],fcycloneinfo[:,5])
				map.plot(x,y,'k',linewidth=3,zorder=2)
				# plot each cyclone with the proper colour scale associated with the intensity
				for i in range(1,fcycloneinfo.shape[0]-1):
					ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
					x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
					plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
					# plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=3)
					plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor=color[ind], linewidth=linwtype[ind], facecolor=color[ind],zorder=3)

				i=0 # first date (+)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='+', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				# cdatein = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				cdatein = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatein, color='m', fontsize=9,horizontalalignment='left',weight='bold',zorder=4)

				i=-1 # last date (x)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='x', facecolor='black', linewidth=1, zorder=4)	
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				# cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatefi, color='m', fontsize=9,horizontalalignment='left',weight='bold',zorder=4)
				# plot a small point at the center of the cyclone when entering a new day
				ind = np.where( np.diff(fcycloneinfo[:,2])!=0  )[0]+1
				if np.any(ind)==True:
					for t in range(0,ind.shape[0]):
						if ind[t]!=0 and ind[t]!=fcycloneinfo.shape[0]-1 :
							x, y = map(fcycloneinfo[ind[t],6],fcycloneinfo[ind[t],5])
							plt.scatter(x,y,marker='.', edgecolor='black', linewidth=1, zorder=4)
							aday=str(int(fcycloneinfo[ind[t],2])).zfill(2)
							plt.text(x,y*(1.+(np.sign(y)*0.04)),aday, color='m',fontsize=10,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)

				# color bar of the cyclone wind scale clevels
				fm = np.zeros((lat.shape[0],lon.shape[0]),'f')-1
				map.contourf(xx,yy,fm,clevels,colors=color,extend="max")
				ax = plt.gca(); pos = ax.get_position()
				l, b, w, h = pos.bounds; cax = plt.axes([1.-l-(db/3.), b+0.05,0.02, h-0.1])
				cbar2=plt.colorbar(cax=cax, orientation='vertical',format='%4.0f',ticks=clevels)
				plt.axes(ax);  cbar2.set_label('Cyclonic Wind scale')

				plt.title(figinfo[4]+'  Cyclone '+cdatein+' (+) to '+cdatefi+' (x)')

			else:
				plt.title(figinfo[4])

			amap = np.append(amap,map); # concatenate Basemaps of each figure (each cyclonpe)

		else:
			# Regional, Equidistant Cylindrical Projection ------------------

			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )
			# Basemap resolution
			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 300:
				fres='i'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 10:
				fres='h'

			map = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution=fres)

			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			# check if background field exists and is different than zero
			if min(np.atleast_2d(bgfield).shape) > 1:
				if bgfieldp[np.isnan(bgfieldp)==False].mean() != 0.0:

					if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100 and lon[-1]<=180:
						map.bluemarble(scale=np.tanh(5000./((lon[-1]-lon[0])*(lat[-1]-lat[0])))); map.drawlsmask(ocean_color='w',land_color='None')

					map.contourf(xx,yy,bgfieldp,figinfo[1],cmap=paletteg,extend="max")

					ax = plt.gca(); pos = ax.get_position()
					l, b, w, h = pos.bounds; cax = plt.axes([l+0.07, b-db, w-0.15, 0.025]) # setup colorbar axes.
					cbar=plt.colorbar(cax=cax, orientation='horizontal',format=nformat); #cbar.ax.tick_params(labelsize=fsmpb)
					tick_locator = ticker.MaxNLocator(nbins=7); cbar.locator = tick_locator; cbar.update_ticks()
					plt.axes(ax); cbar.set_label('Background field scale')

					map.contour(xx,yy,bgfieldp,cmap=paletteg)

				else:
					map.fillcontinents(color='grey')
			else:
				map.fillcontinents(color='grey')

			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.drawmeridians(np.arange(round(lon[np.isnan(lon)==False].min()),round(lon[np.isnan(lon)==False].max()),lonmd),labels=[0,0,0,1])
			map.drawparallels(np.arange(round(lat[np.isnan(lat)==False].min()),round(lat[np.isnan(lat)==False].max()),latmd),labels=[1,0,0,0])

			# check if cycloneinfo was entered
			if np.all(fcycloneinfo==0) == False:
				# marker size
				sizemark=sizemarko*ms
				# continuous black line linking cyclones (o)
				x, y = map(fcycloneinfo[:,6],fcycloneinfo[:,5])
				map.plot(x,y,'k',linewidth=3,zorder=2)
				# plot each cyclone with the proper colour scale associated with the intensity
				for i in range(1,fcycloneinfo.shape[0]-1):
					ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
					x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
					plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
					# plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=3)
					plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor=color[ind], linewidth=linwtype[ind], facecolor=color[ind],zorder=3)

				i=0 # first date (+)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='+', facecolor='black', linewidth=1, zorder=4)
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				# cdatein = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				cdatein = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatein, color='m', fontsize=9,horizontalalignment='left',weight='bold',zorder=4)

				i=-1 # last date (x)
				x, y = map(fcycloneinfo[i,6],fcycloneinfo[i,5])
				ind = int(np.where( abs(np.diff(np.sign(fcycloneinfo[i,-1]-np.append(clevels,np.array([1000]))))) != 0 )[0].max())
				plt.scatter(x,y, s=sizemark[ind]*1.5, marker='o', edgecolor='white', linewidth=linwtype[ind], facecolor='white',zorder=1)
				plt.scatter(x,y, s=sizemark[ind], marker='o', edgecolor='black', linewidth=linwtype[ind], facecolor=color[ind],zorder=4)
				plt.scatter(x,y,s=sizemark[ind],marker='x', facecolor='black', linewidth=1, zorder=4)	
				tdate = time.mktime((np.int(fcycloneinfo[i,0]),np.int(fcycloneinfo[i,1]),np.int(fcycloneinfo[i,2]),np.int(fcycloneinfo[i,3]), np.int(fcycloneinfo[i,4]), 0, 0, 0, 0))
				# cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)+'/'+time.strftime("%b", time.gmtime(tdate))
				cdatefi = str(int(fcycloneinfo[i,2])).zfill(2)
				plt.text(x,y*(1.+(np.sign(y)*0.04)), cdatefi, color='m', fontsize=9,horizontalalignment='left',weight='bold',zorder=4)
				# plot a small point at the center of the cyclone when entering a new day
				ind = np.where( np.diff(fcycloneinfo[:,2])!=0  )[0]+1
				if np.any(ind)==True:
					for t in range(0,ind.shape[0]):
						if ind[t]!=0 and ind[t]!=fcycloneinfo.shape[0]-1 :
							x, y = map(fcycloneinfo[ind[t],6],fcycloneinfo[ind[t],5])
							plt.scatter(x,y,marker='.', edgecolor='black', linewidth=1, zorder=4)
							aday=str(int(fcycloneinfo[ind[t],2])).zfill(2)
							plt.text(x,y*(1.+(np.sign(y)*0.04)),aday, color='m',fontsize=10,verticalalignment='bottom',horizontalalignment='left',weight='bold',zorder=6)

				# color bar of the cyclone wind scale clevels
				fm = np.zeros((lat.shape[0],lon.shape[0]),'f')-1
				map.contourf(xx,yy,fm,clevels,colors=color,extend="max")
				ax = plt.gca(); pos = ax.get_position()
				l, b, w, h = pos.bounds; cax = plt.axes([1.-l-(db/3.), b+0.05,0.02, h-0.1])
				cbar2=plt.colorbar(cax=cax, orientation='vertical',format='%4.0f',ticks=clevels)
				plt.axes(ax);  cbar2.set_label('Cyclonic Wind scale')


				plt.title(figinfo[4]+'  Cyclone '+cdatein+' (+) to '+cdatefi+' (x)')

			else:
				plt.title(figinfo[4])

			amap = np.append(amap,map); # concatenate Basemaps of each figure (each cyclonpe)

		plt.show()

	return amap

def cplotbm(*args):
	'''
	Function to plot cyclone tracks using python basemap with bluemarble background. Developed to use results from cyclone.linktime.

	cyclone.cplotbm(fcyclones,lonlat,figinfo)

	2 or 3 inputs are requested:
	fcyclones : array resulted from cyclone.linktime, containing position of cyclones. 
		    fcyclones[:,0] time ID, not important here
		    fcyclones[:,1] latitude of each cyclone at a specific instant.
		    fcyclones[:,2] longitude of each cyclone at a specific instant.
		    fcyclones[:,3] cyclone ID, linking the same cyclone at different time.

	lonlat : vector with initial and final longitude and latitude
		 For Ex: lonlat=[-180,180,-90,90]
			 lonlat=[0,360,-90,90]
			 lonlat=[20,120,-80,20]
		 lonlat here does not need to be the same as in forecastools.getnoaaforecast. You can select any intervall to zoom-in or zoom-out your cyclone plot.

	figinfo : figure size for Basemap plot
		  For Ex: figinfo=(8,5)
		  If you are using forecastools.plotforecast then you can enter the same figinfo and this code cyclone.cplot will use figinfo[0] only (figure size)
		  figinfo can be omitted or set as zero. In this case, the default size will be used.

	Output consists of a figure with cyclone plots in red and bluemarble background.
		If no cyclones are plotted check fcyclones array, it might be empty and no cyclone was found.
		Figure can be saved using:
        	savefig('NameOfInterest.jpg', dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='jpg',transparent=False, bbox_inches=None, pad_inches=0.1)
		see other options of savefig

	Examples :

		from time import gmtime, strftime, time
		import forecastools
		import cyclone
		from pylab import *
		lonlat=[-80, 20, 0, 85]; nt=57; 
		aux=strftime("%Y-%m-%d %H:%M:%S", gmtime()); initime=[int(aux[0:4]),int(aux[5:7]),int(aux[8:10])]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrd10m']
		[ugrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrd10m']
		[vgrd10m,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','prmslmsl']
		[prmslmsl,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat)
		levels=[850,700]
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','ugrdprs']
		[ugrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','vgrdprs']
		[vgrdprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels[0])
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','absvprs']
		[absvprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		nomads=['http://nomads.ncep.noaa.gov:9090/dods','gfs_0p25','hgtprs']
		[hgtprs,lon,lat,ntime]=forecastools.getnoaaforecast(nomads,initime,nt,lonlat,levels)
		swlat=500.; swlon=500.; sw=np.array([swlat,swlon])
		threshold=[1030.,5./100000.,2./100000.,1500.,3050.]
		candidates=cyclone.findcandidates(lat,lon,prmslmsl,absvprs,hgtprs,threshold,sw)
		restriction=[5,0.0001,0.001] # restriction=[12.86,(1./333.),3]
		[fcandidates,cyclpos] = cyclone.position(lat,lon,candidates,prmslmsl,ugrdprs,vgrdprs,ugrd10m,vgrd10m,restriction,sw)
		maxspeed=60.*0.514; maxhdur=6; mindur=24
		fcyclones = cyclone.linktime(cyclpos,ntime,maxspeed,maxhdur,mindur)
		figinfo=(8,5)
		cyclone.cplotbm(fcyclones,lonlat,figinfo)

	version 1.1:    03/11/2015
	version 1.2:    08/04/2016
	version 1.3:    08/05/2016
	www.atmosmarine.com
	'''

	if len(args) < 2:
		sys.exit(' ERROR! Insufficient input arguments. At least fcyclones and lonlat must be provided')
	elif len(args) == 2:
		fcyclones=copy.copy(args[0]); lonlat=copy.copy(args[1]); figinfo=0
	elif len(args) == 3:
		fcyclones=copy.copy(args[0]); lonlat=copy.copy(args[1]); figinfo=copy.copy(args[2])
	else:
		sys.exit(' ERROR! Too many arguments. fcyclones, lonlat and figinfo are necessary only')

	if len(fcyclones) == 0 :
		print(' ERROR! No cyclones in fcyclones, it is empty')


	# Artificial lon lat array just to plot with Basemap
	lon=np.linspace(lonlat[0],lonlat[1],101)
	lat=np.linspace(lonlat[2],lonlat[3],101)

	# opening figure with figure size in case it is informed  
	if figinfo!=0:
		if len(figinfo)>2:
			fig=plt.figure(figsize=figinfo[0])
		else:
			fig=plt.figure(figsize=figinfo)
	else:
		fig=plt.figure()
	# PLOTS
	if abs(lonlat[0]-lonlat[1])>350:
			# Global Plot with Eckert IV Projection	
			map = Basemap(projection='eck4',lon_0 = 0, resolution = 'l')
			map.bluemarble(scale=0.2) # map.shadedrelief() map.etopo()
			#map.drawmapboundary(fill_color='blue')
			#map.fillcontinents(color='coral',lake_color='aqua')
			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.drawmeridians(np.arange(round(lon.min()),round(lon.max()),40),labels=[0,0,0,1],linewidth=0.3,fontsize=6)
			map.drawparallels(np.arange(round(lat.min()),round(lat.max()),20),labels=[1,0,0,0],linewidth=0.3,fontsize=6)
			if len(fcyclones) != 0 :
				for c in range(int(fcyclones[:,3].min()),int(fcyclones[:,3].max())+1):
					a = int(np.where(fcyclones[:,3]==c)[0])
					cycl=np.copy(fcyclones[a,1:-1])
					x, y = map(cycl[:,1],cycl[:,0])
					plt.plot(x,y,'r.-')
			plt.show()

	elif abs(lonlat[2]-lonlat[3])>20 and abs(lonlat[2]-lonlat[3])<70 and abs(np.array(lonlat[2::])).max()>60 and (float(lonlat[1]-lonlat[0])/float(lonlat[3]-lonlat[2]))<3. and (lonlat[1]-lonlat[0])<60:
			# Regional High latitudes, Stereographic Projection ------------------
			# size/shape of the figure
			nwidth=int( (5000000*(lon[-1]-lon[0]))/50 )
			nheight=int( (5000000*(lat[-1]-lat[0]))/40 )
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )

			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 300:
				fres='i'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 10:
				fres='h'

			map = Basemap(width=nwidth, height=nheight, resolution=fres, projection='stere',
				lat_ts=lat[0], lat_0=((lat[-1]-lat[0])/2)+lat[0], lon_0=((lon[-1]-lon[0])/2)+lon[0] )

			map.bluemarble(scale=np.tanh(4000./((lon[-1]-lon[0])*(lat[-1]-lat[0]))))
			#map.drawmapboundary(fill_color='blue')
			#map.fillcontinents(color='coral',lake_color='aqua')     
			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			map.drawcoastlines(linewidth=0.8)
			map.drawcountries(linewidth=0.5)
			map.drawmeridians(np.arange(round(lon[0]),round(lon[-1]),lonmd),labels=[0,0,0,1],linewidth=0.3,fontsize=8)
			map.drawparallels(np.arange(round(lat[0]),round(lat[-1]),latmd),labels=[1,0,0,0],linewidth=0.3,fontsize=8)
			if len(fcyclones) != 0 :
				for c in range(int(fcyclones[:,3].min()),int(fcyclones[:,3].max())+1):
					a = int(np.where(fcyclones[:,3]==c)[0])
					cycl=np.copy(fcyclones[a,1:-1])
					x, y = map(cycl[:,1],cycl[:,0])
					plt.plot(x,y,'r.-')
			plt.show()

	else:
			# Regional, Equidistant Cylindrical Projection ------------------
			# lat and lon parallels and meridians displacement
			lonmd=int( (lon[np.isnan(lon)==False].max()-lon[np.isnan(lon)==False].min()) / 4 )
			latmd=int( (lat[np.isnan(lat)==False].max()-lat[np.isnan(lat)==False].min()) / 4 )
			if lat[0] < 0 and lat[-1] <= 0:
				lat_0=-(abs(lat[-1])+abs(lat[0]))/2.0
			else:
				lat_0=(lat[0]+lat[-1])/2.0

			if lon[0] < 0 and lon[-1] <= 0:
				lon_0=-(abs(lon[-1])+abs(lon[0]))/2.0
			else:
				lon_0=(lon[0]+lon[-1])/2.0

			#Equidistant Cylindrical Projection

			fres='l'
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 300:
				fres='i'
			elif ((lon[-1]-lon[0])*(lat[-1]-lat[0])) < 10:
				fres='h'

			map = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution=fres)
			if ((lon[-1]-lon[0])*(lat[-1]-lat[0])) > 100 and lon[-1]<=180:
				map.bluemarble(scale=np.tanh(5000./((lon[-1]-lon[0])*(lat[-1]-lat[0]))))
			else:
				map.drawmapboundary(fill_color='blue')
				map.drawcoastlines(linewidth=0.8)
				map.drawcountries(linewidth=0.5) 
				map.fillcontinents(color='0.2',lake_color='0.5')

			[mnlon,mnlat]=np.meshgrid(lon[:],lat[:])
			xx, yy = map(mnlon,mnlat)
			#map.drawcoastlines(linewidth=0.8)
			#map.drawcountries(linewidth=0.5)  
			map.drawmeridians(np.arange(round(lon.min()),round(lon.max()),lonmd),labels=[0,0,0,1])
			map.drawparallels(np.arange(round(lat.min()),round(lat.max()),latmd),labels=[1,0,0,0])
			if len(fcyclones) != 0 :
				for c in range(int(fcyclones[:,3].min()),int(fcyclones[:,3].max())+1):
					a = int(np.where(fcyclones[:,3]==c)[0])
					cycl=np.copy(fcyclones[a,1:-1])
					x, y = map(cycl[:,1],cycl[:,0])
					plt.plot(x,y,'r.-')
			plt.show()


