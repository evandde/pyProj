from matplotlib import pyplot as plt 
import numpy as np  
from scipy import signal
import lib.dataRefinement as dr

rawdataDir = "/home/yk/research/20200800_GammaDensitometer/rawdata/"

filename1 = "20200814_Simul_Air_1.txt"
rawdata1 = np.loadtxt(rawdataDir + filename1, skiprows=1)
rawEDep = rawdata1[:, 1]
eDep = dr.ApplyEnergyResolution(rawEDep, percentER662=7.)

filename2 = "20200813_Air_1.tka"
expdata = np.loadtxt(rawdataDir + filename2)

hist,bins = np.histogram(eDep, bins=np.arange(0., 1.5, 0.01)) 
# plt.title("histogram") 
# plt.plot(bins[:-1], hist)
# plt.show()

plt.plot(bins[:-1], hist)
plt.draw()
pts = plt.ginput(n=2)
xIdx = np.digitize(np.array([pts[0][0], pts[1][0]]), bins)
peaks, _ = signal.find_peaks(hist[xIdx[0]:xIdx[1]], prominence=100)
peaks += xIdx[0]

plt.plot(bins[peaks], hist[peaks], "xr")
plt.draw()

calibPts = []
for i in range(len(peaks)):
    ene = float(input("Peak energy (MeV): "))
    calibPts.append((peaks[i], ene))
# print(pt)

plt.show()
