import os
import numpy as np
import multiprocessing

from os import listdir
from os.path import isfile,join

root = "/run/media/sagauga/data/data/jpas/mini_jpas_1488-11754/1488-11754/"

p   = root+"original_stamps/"

sample = np.loadtxt(p+"filelist",dtype='str',usecols=(0),comments="#", delimiter=",", unpack=False)

# list all files in a folder and set it to the array sample.
# sample = np.array([f for f in listdir(p) if isfile(join(p, f))])

print(sample)

def run(file_name):
    root3 = "/run/media/sagauga/wall/splus_cuts/gals_r_petro_up_to-17_prob_06_new_table/"
    psf = root+"psf/psf_"

    print(file_name)

    root2 = "/run/media/sagauga/data/"
    # os.system("python3 "+root2+"GoogleDrive/research/codes/galclean.py "+p+file_name+" --save --siglevel 10.5 --min_size 0.008")
    os.system("python3 "+root2+"GoogleDrive/research/codes/morfometryka80.py "+p+file_name+" "+psf+file_name.replace(".fits",".fits")+" noshow profile stangal clean")
    # os.system("python3 "+root2+"GoogleDrive/research/codes/mfmtk_isophote.py "+p+file_name+" "+psf+file_name.replace(band+"_seg.fits","R.fits")+"")# galpetro galnostars")
        # os.system("python3 "+root2+"GoogleDrive/research/codes/morfometryka72_3.py "+p+file_name.replace(".fits","")+"_seg.fits"+" "+psf+" rerun noshow profile galpetro galnostars")
    # os.system("python3 "+root2+"GoogleDrive/research/codes/morfometryka80.py "+p+file_name.replace(".fits","")+".fits"+" "+psf+file_name.replace("_seg.fits",".fits")+ " rerun noshow profile")
    # os.system("python3 /run/media/sagauga/ssd_files/GoogleDrive/research/codes/kurvature_sagv4_py3.py --f "+p+file_name+" --p "+psf+ "")
    # os.system("python3 /run/media/sagauga/data/GoogleDrive/GitLab/kurvature/kurvature.py --pr "+root3+file_name.replace(".fits","") + "_IR.csv --mpr "+ p+file_name.replace(".fits","")+"_profiles.csv")#+" --p "+p+"psf/psf_"+file_name) # psf_efigi_s13.fits
    # os.system("python3 /run/media/sagauga/data/GoogleDrive/GitLab/kurvature/kurvature.py --pr "+p+file_name.replace(".fits","") + "_IR.csv --mpr "+ p+file_name.replace(".fits","")+"_profiles.csv")#+" --p "+p+"psf/psf_"+file_name) # psf_efigi_s13.fits
    return file_name

NProc = 6 #the number of your computer threads.
pool = multiprocessing.Pool( NProc )
tasks = []
# for SAMPLE in sample:#13296
for SAMPLE in sample:
	tasks.append((SAMPLE,))

results = [pool.apply_async( run, t ) for t in tasks]
for result in results:
	Filename = result.get()
pool.close()
pool.join()
