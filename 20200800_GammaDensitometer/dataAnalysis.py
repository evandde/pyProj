import matplotlib.pyplot as plt
import numpy as np
import evanLib.dataRefinement as dr

rawdataDir = "/home/yk/research/20200800_GammaDensitometer/rawdata/"
plt.rc('font', size=22, weight='bold')
plt.rc('legend', fontsize=16)
plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')

# experimental data
# 1. calibration
ch = np.arange(2048)
# calibFilename = "20200813_CalibCsCo_10min.TKA"
calibFilename = "20200819_CalibCsCo_10min.TKA"
# calibFilename = "20200908_CalibCs137Co60_600s.TKA"
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
print(np.sum(bkgdata))
# plt.plot(eneAxis, bkgdata/3.)
expFilename = "20200819_Air_10min_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
airdata = expdata - bkgdata
print(np.sum(airdata))
# plt.plot(eneAxis, airdata)
expFilename = "20200908_Wat100_600s_Off1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
watdata = expdata - bkgdata
expFilename = "20200908_Wat100_600s_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
watdata2 = expdata - bkgdata
# plt.plot(eneAxis, watdata)
expFilename = "20200908_Con100_600s_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
condata1 = expdata - bkgdata
expFilename = "20200908_Con100_600s_2.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
condata2 = expdata - bkgdata
expFilename = "20200908_Con100_600s_3.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
condata3 = expdata - bkgdata
condata = (condata1 + condata2 + condata3) / 3.
# plt.plot(eneAxis, condata)
expFilename = "20200908_W100C15_600s_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
con15data1 = expdata - bkgdata
expFilename = "20200908_W100C15_600s_2.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
con15data2 = expdata - bkgdata
expFilename = "20200908_W100C15_600s_3.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
con15data3 = expdata - bkgdata
con15data = (con15data1 + con15data2 + con15data3) / 3.
# plt.plot(eneAxis, condata)
expFilename = "20200908_W100C30_600s_1.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
con30data1 = expdata - bkgdata
expFilename = "20200908_W100C30_600s_2.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
con30data2 = expdata - bkgdata
expFilename = "20200908_W100C30_600s_3.TKA"
expdata = np.loadtxt(rawdataDir + expFilename) / (600)
con30data3 = expdata - bkgdata
con30data = (con30data1 + con30data2 + con30data3) / 3.
# plt.plot(eneAxis, condata)

# plt.legend(["WATER", "CONCRETE"])
# plt.xlabel("Energy (MeV)")
# plt.ylabel("Counts/ch/s")
# plt.show()

# 3. simulation data
simulFilename1 = "20200827_Simul_Air_1.txt"
rawdata = np.loadtxt(rawdataDir + simulFilename1, skiprows=1)
rawEDep = rawdata[:, 1]
eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
hist, _ = np.histogram(eDep, bins=eneAxis)
airdataSim = np.append(hist / (10*60), 0.)
print(np.sum(airdataSim))
# plt.plot(eneAxis, airdataSim)
# plt.show()

# simulFilename1q = "20200831_Simul_Air_1_QGSPBERT.txt"
# rawdata = np.loadtxt(rawdataDir + simulFilename1q, skiprows=1)
# rawEDep = rawdata[:, 1]
# eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
# hist, _ = np.histogram(eDep, bins=eneAxis)
# airdataSim2 = np.append(hist / (10*60), 0.)
# # plt.plot(eneAxis, airdataSim2)
# # plt.show()

# simulFilename2 = "20200827_Simul_Wat100_1.txt"
# rawdata = np.loadtxt(rawdataDir + simulFilename2, skiprows=1)
# rawEDep = rawdata[:, 1]
# eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
# hist, _ = np.histogram(eDep, bins=eneAxis)
# watdataSim = np.append(hist / (10*60), 0.)
# # plt.plot(eneAxis, watdataSim)
# # plt.show()

# simulFilename3 = "20200827_Simul_Con100_1.txt"
# rawdata = np.loadtxt(rawdataDir + simulFilename3, skiprows=1)
# rawEDep = rawdata[:, 1]
# eDep = dr.ApplyEnergyResolution(rawEDep, a=erCoeff)
# hist, _ = np.histogram(eDep, bins=eneAxis)
# condataSim = np.append(hist / (10*60), 0.)
# # plt.plot(eneAxis, condataSim)
# # plt.show()

# 4. Analyze
# plt.plot(eneAxis, bkgdata)
# plt.plot(eneAxis, bkgdata + airdata)
# plt.title("Air")
# plt.legend(["Bkg", "Air"])
# plt.show()

plt.plot(eneAxis, airdata)
plt.plot(eneAxis, airdataSim)
# plt.plot(eneAxis, airdataSim2)
plt.title("Air")
plt.legend(["Exp.", "Simul. FTFP"])
plt.show()

# plt.plot(eneAxis, watdata)
# plt.plot(eneAxis, watdata2)
# plt.title("Water")
# plt.legend(["Motor on", "Motor off"])
# plt.show()

# plt.plot(eneAxis, watdata)
# plt.plot(eneAxis, condata)
# plt.plot(eneAxis, con15data)
# plt.plot(eneAxis, con30data)
# plt.title("Comparison")
# plt.legend(["Water", "Concrete", "W:C=100:15", "W:C=100:30"])
# plt.show()
# plt.savefig("test.tif", dpi=300)

# areaWater, areaErrWater, _, _ = dr.AnalyzePeaks(eneAxis, watdata, nPeaks=2)
# # print([areaWater, areaErrWater])
# print(dr.MulSumRatioOf2Peaks(areaWater))
# areaWaterSim, areaErrWaterSim, _, _ = dr.AnalyzePeaks(eneAxis, watdataSim, nPeaks=2)
# # print([areaWater, areaErrWater])
# print(dr.MulSumRatioOf2Peaks(areaWaterSim))
#
# areaConcrete, areaErrConcrete, _, _ = dr.AnalyzePeaks(eneAxis, watdata, nPeaks=2)
# print([areaConcrete, areaErrConcrete])
# print(dr.MulSumRatioOf2Peaks(areaConcrete))
# areaConcrete, areaErrConcrete, _, _ = dr.AnalyzePeaks(eneAxis, con15data, nPeaks=2)
# print([areaConcrete, areaErrConcrete])
# print(dr.MulSumRatioOf2Peaks(areaConcrete))
# areaConcrete, areaErrConcrete, _, _ = dr.AnalyzePeaks(eneAxis, con30data, nPeaks=2)
# print([areaConcrete, areaErrConcrete])
# print(dr.MulSumRatioOf2Peaks(areaConcrete))
# areaConcrete, areaErrConcrete, _, _ = dr.AnalyzePeaks(eneAxis, condata, nPeaks=2)
# print([areaConcrete, areaErrConcrete])
# print(dr.MulSumRatioOf2Peaks(areaConcrete))
# areaConcreteSim, areaErrConcreteSim, _, _ = dr.AnalyzePeaks(eneAxis, condataSim, nPeaks=2)
# # print([areaConcrete, areaErrConcrete])
# print(dr.MulSumRatioOf2Peaks(areaConcreteSim))

