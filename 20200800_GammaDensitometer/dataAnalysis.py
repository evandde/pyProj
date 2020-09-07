import matplotlib.pyplot as plt
import numpy as np
import lib.dataRefinement as dr

rawdataDir = "/home/yk/research/20200800_GammaDensitometer/rawdata/"
plt.rc('font', size=22, weight='bold')

# experimental data
# 1. calibration
ch = np.arange(2048)
calibFilename = "20200813_CalibCsCo_10min.TKA"
# calibFilename = "20200819_CalibCsCo_10min.TKA"
calibData = np.loadtxt(rawdataDir + calibFilename) / (10*60)

eneCalibModel = dr.EnergyCalibration(ch, calibData)
eneAxis = eneCalibModel.predict([[i] for i in ch])
np.savetxt(rawdataDir + "eneAxis.txt", eneAxis)
# eneAxis = np.loadtxt(rawdataDir + "eneAxis.txt")

# fwhms = dr.FWHMofPeak(eneAxis, calibData)
# print(fwhms)
areaCs137, areaErrCs137, fwhmCs137, pars = dr.AnalyzePeaks(eneAxis, calibData)
areaCo60, areaErrCo60, fwhmCo60, pars = dr.AnalyzePeaks(eneAxis, calibData, nPeaks=2)
fwhms = [fwhmCs137, *fwhmCo60]
erCoeff = np.linalg.lstsq(np.sqrt(np.array([[0.662], [1.173], [1.332]])), fwhms, rcond=None)[0]
print(erCoeff)
# erCoeff = 0.0579

# # 2. exp data
expFilename = "20200819_Bkg_30min.TKA"
bkgdata = np.loadtxt(rawdataDir + expFilename) / (30*60)
# plt.plot(eneAxis, bkgdata/3.)
# expFilename = "20200819_Air_10min_2.TKA"
# expdata = np.loadtxt(rawdataDir + expFilename) / (10*60)
# airdata = expdata - bkgdata
# plt.plot(eneAxis, airdata)
expFilename = "20200819_Wat100_10min_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (10*60)
watdata = expdata - bkgdata
# plt.plot(eneAxis, watdata)
expFilename = "20200819_Con100_10min_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (10*60)
condata = expdata - bkgdata
# plt.plot(eneAxis, condata)

# plt.legend(["WATER", "CONCRETE"])
# plt.xlabel("Energy (MeV)")
# plt.ylabel("Counts/ch/s")
# plt.show()

# 3. simulation data
# simulFilename1 = "20200827_Simul_Air_1.txt"
# rawdata = np.loadtxt(rawdataDir + simulFilename1, skiprows=1)
# rawEDep = rawdata[:, 1]
# eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
# hist, bins = np.histogram(eDep, bins=eneCalibModel.predict([[i] for i in np.arange(2049)]))
# plt.plot(bins[:-1], hist)

simulFilename2 = "20200827_Simul_Wat100_1.txt"
rawdata = np.loadtxt(rawdataDir + simulFilename2, skiprows=1)
rawEDep = rawdata[:, 1]
eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
hist, _ = np.histogram(eDep, bins=eneAxis)
watdataSim = np.append(hist / (10*60), 0.)
# plt.plot(eneAxis, watdataSim)
# plt.show()

simulFilename3 = "20200827_Simul_Con100_1.txt"
rawdata = np.loadtxt(rawdataDir + simulFilename3, skiprows=1)
rawEDep = rawdata[:, 1]
eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
hist, _ = np.histogram(eDep, bins=eneAxis)
condataSim = np.append(hist / (10*60), 0.)
# plt.plot(eneAxis, condataSim)
# plt.show()

# 4. Analyze
plt.plot(eneAxis, watdata)
plt.plot(eneAxis, watdataSim)
plt.title("Water")
plt.legend(["Exp.", "Simul."])
plt.show()

plt.plot(eneAxis, condata)
plt.plot(eneAxis, condataSim)
plt.title("Concrete")
plt.legend(["Exp.", "Simul."])
plt.show()

areaWater, areaErrWater, _, _ = dr.AnalyzePeaks(eneAxis, watdata, nPeaks=2)
# print([areaWater, areaErrWater])
print(dr.MulSumRatioOf2Peaks(areaWater))
areaWaterSim, areaErrWaterSim, _, _ = dr.AnalyzePeaks(eneAxis, watdataSim, nPeaks=2)
# print([areaWater, areaErrWater])
print(dr.MulSumRatioOf2Peaks(areaWaterSim))

areaConcrete, areaErrConcrete, _, _ = dr.AnalyzePeaks(eneAxis, condata, nPeaks=2)
# print([areaConcrete, areaErrConcrete])
print(dr.MulSumRatioOf2Peaks(areaConcrete))
areaConcreteSim, areaErrConcreteSim, _, _ = dr.AnalyzePeaks(eneAxis, condataSim, nPeaks=2)
# print([areaConcrete, areaErrConcrete])
print(dr.MulSumRatioOf2Peaks(areaConcreteSim))