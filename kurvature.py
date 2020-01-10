#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Codiname + <<version>>: Sagauga <<v3>> -- 11/2019
Geferson Lucatelli

|  /  |	   |  |----|   \	  /    /\   ----|---- |	   |  |----|  |-----
| /   |	   |  |____|	\	 /	  /  \		|     |	   |  |____|  |___
| \   |	   |  | * \ 	 \	/	 / __ \ 	|     |	   |  | * \   |
|  \  |____|  |	   \	  \/	/	   \    |	  |____|  |	   \  |____
"""
########################################
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as si
import scipy.signal
import argparse
import os
from matplotlib import gridspec
import pylab as pl
from matplotlib import cm
import astropy.io.fits as pf
import scipy.ndimage as snd
from medpy.filter.smoothing import anisotropic_diffusion as AD
