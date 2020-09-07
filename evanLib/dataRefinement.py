import numpy as np
import scipy.signal
import scipy.optimize
from sklearn import linear_model
import matplotlib.pyplot as plt


def ApplyEnergyResolution(energy:np.ndarray, percentER662:float = -1., a:float = -1.):
    # FWHM=a*sqrt(E)
    if percentER662>=0. and a>=0.:
        print("Both 'a' and 'percentER662' are provided.\n 'a' will be ignored.\n")
    if percentER662>=0.:
        a = percentER662/100.*np.sqrt(0.662)
    
    fwhm = a*np.sqrt(energy)
    energyNew = np.random.normal(energy, fwhm/2.35)
    
    return energyNew


def EnergyCalibration(x:np.ndarray, cnt:list):
    if type(cnt) is np.ndarray:
        cnt = [cnt]

    calibPts = []
    enes = [0.662, 1.173, 1.332]
    k = 0
    
    for n in range(len(cnt)):
        y = cnt[n]
        plt.plot(x, y)
        plt.draw()

        pts = plt.ginput(n=2)
        xIdx = np.digitize(np.array([pts[0][0], pts[1][0]]), x)
        peaks, _ = scipy.signal.find_peaks(y[xIdx[0]:xIdx[1]], prominence=np.max(cnt)*0.1)
        peaks += xIdx[0]
        plt.plot(x[peaks], y[peaks], "xr")
        plt.draw()
        plt.show()

        for i in range(len(peaks)):
            # ene = float(input("Peak energy (MeV): ")); enes.append(ene)
            ene = enes[k]; k += 1
            calibPts.append((peaks[i], ene))
    
    print("-- Selected (channel, energy) points --")
    print(calibPts)
    chPts = [[i[0]] for i in calibPts]
    enePts = [i[1] for i in calibPts]
    plt.plot(chPts, enePts, "xr")
    
    model = linear_model.LinearRegression()
    model.fit(chPts, enePts)
    chtmp = [[i] for i in np.arange(2048)]
    enetmp = model.predict(chtmp)
    plt.plot(chtmp, enetmp)
    plt.show()

    return model


def FWHMofPeak(x:np.ndarray, y:np.ndarray):
    plt.plot(x, y)
    plt.draw()

    pts = plt.ginput(n=2)
    xIdx = np.digitize(np.array([pts[0][0], pts[1][0]]), x)
    peaks, _ = scipy.signal.find_peaks(y[xIdx[0]:xIdx[1]], prominence=np.max(y)*0.1)
    rsltsHalf = scipy.signal.peak_widths(y[xIdx[0]:xIdx[1]], peaks, rel_height=0.5)
    # rsltsFull = scipy.signal.peak_widths(y[xIdx[0]:xIdx[1]], peaks, rel_height=1.)
    #
    peaks += xIdx[0]
    xIdxHalves = np.squeeze(rsltsHalf[2:]) + xIdx[0]
    xHalves = np.interp(xIdxHalves, xIdx, x[xIdx])
    # xIdxFulls = np.squeeze(rslt_full[2:]) + xIdx[0]
    # xFulls = np.interp(xIdxFulls, xIdx, x[xIdx])
    #
    plt.plot(x[peaks], y[peaks], "xr")
    plt.hlines(rsltsHalf[1], *xHalves, color="C2")
    # plt.hlines(rslt_full[1], *xFulls, color="C3")
    plt.show()

    return xHalves[1, :] - xHalves[0, :]


def G1PeakP1Bkg(x, p0, p1, a, b, c):
    return a * np.exp(-(x-b)**2 / (2*(c**2))) + (p0 + p1*x)


def G2PeakP1Bkg(x, p0, p1, a1, b1, c1, a2, b2, c2):
    return a1 * np.exp(-(x-b1)**2 / (2*(c1**2))) + a2 * np.exp(-(x-b2)**2 / (2*(c2**2))) + (p0 + p1*x)


def AnalyzePeaks(x, y, nPeaks=1):
    plt.plot(x, y)
    plt.draw()

    pts = plt.ginput(n=2)

    xIdx = np.digitize(np.array([pts[0][0], pts[1][0]]), x)

    xSel = x[xIdx[0]:xIdx[1]]
    ySel = y[xIdx[0]:xIdx[1]]

    peaks, _ = scipy.signal.find_peaks(ySel, prominence=np.max(y)*0.1)

    if nPeaks == 1:
        pars, cov = scipy.optimize.curve_fit(f=G1PeakP1Bkg, xdata=xSel, ydata=ySel,
                                             p0=[1., -1., ySel[peaks[0]], xSel[peaks[0]], 0.01],
                                             bounds=([0., -np.inf, 0., xSel[0], 0.],
                                                     [np.inf, 0., np.inf, xSel[-1], xSel[-1]-xSel[0]]))
        a = np.sqrt(2 * np.pi) * 0.997
        A = pars[2]
        sA = np.sqrt(cov[2, 2])
        B = pars[4]
        sB = np.sqrt(cov[4, 4])
        sAB = cov[2, 4]
        area = a * A * B
        areaErr = np.sqrt(a * (A * B) ** 2 * ((sA / A) ** 2 + (sB / B) ** 2 + 2 * (sAB / (A * B))))
        fwhm = pars[4] * 2 * np.sqrt(2 * np.log(2))

        plt.plot(xSel, G1PeakP1Bkg(xSel, *pars))
        plt.draw()

    elif nPeaks == 2:
        pars, cov = scipy.optimize.curve_fit(f=G2PeakP1Bkg, xdata=xSel, ydata=ySel,
                                             p0=[1., -1., ySel[peaks[0]], xSel[peaks[0]], 0.01, ySel[peaks[1]], xSel[peaks[1]], 0.01],
                                             bounds=([0., -np.inf, 0., xSel[0], 0., 0., xSel[int(xSel.size/2)], 0.],
                                                     [np.inf, 0., np.inf, xSel[int(xSel.size/2)], (xSel[-1]-xSel[0])/2, np.inf, xSel[-1], (xSel[-1]-xSel[0])/2]))
        a = np.sqrt(2 * np.pi) * 0.997
        A = pars[2]
        sA = np.sqrt(cov[2, 2])
        B = pars[4]
        sB = np.sqrt(cov[4, 4])
        sAB = cov[2, 4]
        area1 = a * A * B
        areaErr1 = np.sqrt(a * (A * B) ** 2 * ((sA / A) ** 2 + (sB / B) ** 2 + 2 * (sAB / (A * B))))
        fwhm1 = pars[4] * 2 * np.sqrt(2 * np.log(2))

        A = pars[5]
        sA = np.sqrt(cov[5, 5])
        B = pars[7]
        sB = np.sqrt(cov[7, 7])
        sAB = cov[5, 7]
        area2 = a * A * B
        areaErr2 = np.sqrt(a * (A * B) ** 2 * ((sA / A) ** 2 + (sB / B) ** 2 + 2 * (sAB / (A * B))))
        fwhm2 = pars[7] * 2 * np.sqrt(2 * np.log(2))

        area = [area1, area2]
        areaErr = [areaErr1, areaErr2]
        fwhm = [fwhm1, fwhm2]

        plt.plot(xSel, G2PeakP1Bkg(xSel, *pars))
        plt.draw()

    plt.show()

    return area, areaErr, fwhm, pars


def MulSumRatioOf2Peaks(areas: np.ndarray):
    i1, i2 = areas
    return (i1 * i2) / (i1 + i2)
