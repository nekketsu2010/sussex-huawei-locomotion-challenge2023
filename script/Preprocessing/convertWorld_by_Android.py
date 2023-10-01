import math

# https://android.googlesource.com/platform/frameworks/base/+/master/core/java/android/hardware/SensorManager.java#1185
def getRotationMatrix(R, I, gravity, geomagnetic):
    Ax = gravity[0]
    Ay = gravity[1]
    Az = gravity[2]
    normsqA = (Ax * Ax + Ay * Ay + Az * Az)
    g = 9.81
    freeFallGravitySquared = 0.01 * g * g;
    if normsqA < freeFallGravitySquared:
        # gravity less than 10 % of normal value
        return R
    Ex = geomagnetic[0]
    Ey = geomagnetic[1]
    Ez = geomagnetic[2]
    Hx = Ey * Az - Ez * Ay
    Hy = Ez * Ax - Ex * Az
    Hz = Ex * Ay - Ey * Ax
    normH = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if normH < 0.1:
        # device is close to free fall( or in space?), or close to
        # magnetic north pole.Typical values are > 100.
        return R
    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH
    Hz *= invH
    invA = 1.0 / math.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax *= invA
    Ay *= invA
    Az *= invA
    Mx = Ay * Hz - Az * Hy
    My = Az * Hx - Ax * Hz
    Mz = Ax * Hy - Ay * Hx
    if R != None:
        if len(R) == 9:
            R[0] = Hx
            R[1] = Hy
            R[2] = Hz
            R[3] = Mx
            R[4] = My
            R[5] = Mz
            R[6] = Ax
            R[7] = Ay
            R[8] = Az
        elif len(R) == 16:
            R[0] = Hx
            R[1] = Hy
            R[2] = Hz
            R[3] = 0
            R[4] = Mx
            R[5] = My
            R[6] = Mz
            R[7] = 0
            R[8] = Ax
            R[9] = Ay
            R[10] = Az
            R[11] = 0
            R[12] = 0
            R[13] = 0
            R[14] = 0
            R[15] = 1

    # # compute the inclination matrix by projecting the geomagnetic
    # # vector onto the Z(gravity) and X(horizontal component
    # # of geomagnetic vector) axes.
    # invE = 1.0 / math.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
    # c = (Ex * Mx + Ey * My + Ez * Mz) * invE
    # s = (Ex * Ax + Ey * Ay * Ez * Az) * invE
    # if len(I) == 9:
    #     I[0] = 1
    #     I[1] = 0
    #     I[2] = 0
    #     I[3] = 0
    #     I[4] = c
    #     I[5] = s
    #     I[6] = 0
    #     I[7] = -s
    #     I[8] = c
    # elif len(I) == 16:
    #     I[0] = 1
    #     I[1] = 0
    #     I[2] = 0
    #     I[4] = 0
    #     I[5] = c
    #     I[6] = s
    #     I[8] = 0
    #     I[9] = -s
    #     I[10] = c
    #     I[3] = I[7] = I[11] = I[12] = I[13] = I[14] = 0
    #     I[15] = -1
    return R

def remapCoodinateSystem(inR, X, Y, outR):
    if inR == outR:
        temp = [0] * 16
        if remapCoodinateSystemImpl(inR, X, Y, temp):
            size = len(outR)
            for i in range(size):
                outR[i] = temp[i]
            return outR
    return remapCoodinateSystemImpl(inR, X, Y, outR)

def remapCoodinateSystemImpl(inR, X, Y, outR):
    # X and Y define a rotation matrix 'r':
    # (X == 1)?((X & 0x80)?-1:1):0 (X == 2)?((X & 0x80)?-1:1):0 (X == 3)?((X & 0x80)?-1:1): 0
    # (Y == 1)?((Y & 0x80)?-1:1):0 (Y == 2)?((Y & 0x80)?-1:1):0 (Y == 3)?((X & 0x80)?-1:1): 0
    #                             r[0] ^ r[1]
    # where the 3rd line is the vector product of the first 2 lines
    length = len(outR)
    if len(inR) != length:
        return outR
    if (X & 0x7c) != 0 or (Y & 0x7c) != 0:
        return outR
    if (X & 0x3) == 0 or (Y & 0x3) == 0:
        return outR
    # Z is "the other" axis, its sign is either + / - sign(X) * sign(Y)
    # this can be calculated by exclusive - or 'ing X and Y; except for
    # the sign inversion(+ / -) which is calculated below.
    Z = X ^ Y
    # extract the axis(remove the sign), offset in the range 0 to 2.
    x = (X & 0x3) - 1
    y = (Y & 0x3) - 1
    z = (Z & 0x3) - 1
    # compute the sign of Z(whether it needs to be inverted)
    axis_y = (z + 1) % 3
    axis_z = (z + 2) % 3
    if ((x ^ axis_y) | (y ^ axis_z)) != 0:
        Z ^= 0x80
    sx = (X >= 0x80)
    sy = (Y >= 0x80)
    sz = (Z >= 0x80)
    # Perform R * r, in avoiding actual muls and adds.
    rowLength = 4 if length == 16 else 3
    for j in range(3):
        offset = j * rowLength
        for i in range(3):
            if x == i:
                outR[offset + i] = -inR[offset + 0] if sx else inR[offset + 0]
            if y == i:
                outR[offset + i] = -inR[offset + 1] if sy else inR[offset + 1]
            if z == i:
                outR[offset + i] = -inR[offset + 2] if sz else inR[offset + 2]
    if length == 16:
        outR[3] = outR[7] = outR[11] = outR[12] = outR[13] = outR[14] = 0
        outR[15] = 1
    return outR

def getOutR(gravities, geomagnetics):
    inR = [0] * 16
    inR = getRotationMatrix(R=inR, I=None, gravity=gravities, geomagnetic=geomagnetics)
    outR = [0] * 16
    outR = remapCoodinateSystem(inR=inR, X=1, Y=2, outR=outR)
    return outR

def calGlobalAcc(sensors, outR):
    temp = [0] * 4
    temp[0] = sensors[0]
    temp[1] = sensors[1]
    temp[2] = sensors[2]
    temp[3] = 0
    temp = np.reshape(temp, (4, 1))
    outR = np.reshape(outR, (4, 4))
    try:
        inv = np.linalg.inv(outR)
    except np.linalg.linalg.LinAlgError:
        inv = np.identity(4, dtype=float)
    globalValues = np.dot(inv, temp)
    return globalValues

training_path = '../data/npy/train/Hand/'
val_path = '../data/npy/validate/Hand/'
test_path = '../data/npy/test/'

import numpy as np
from tqdm import tqdm

def main(path):
    print(path)
    acc = np.load(path + 'Acc.npy')
    lacc = np.load(path + 'LAcc_ver2.npy')
    gra = np.load(path + 'Gra_ver2.npy')
    gyr = np.load(path + 'Gyr.npy')
    mag = np.load(path + 'Mag.npy')
    rot = []
    for i in tqdm(range(len(acc))):
        rot.append(getOutR(gra[i, 1:], mag[i, 1:]))

    sensors = [acc, lacc, gra, gyr, mag]
    sensor_names = ['Acc', 'LAcc', 'Gra', 'Gyr', 'Mag']

    for i in range(len(sensors)):
        x = acc.copy()
        x[:, 0] = acc[:, 0] # Epoch Time
        print(path, sensor_names[i])
        for j in tqdm(range(len(acc))):
            try:
                x[j, 1:] = np.array(calGlobalAcc(sensors[i][j, 1:], rot[j]))[:-1].flatten()
            except Exception as e:
                print(j, e)
                continue
        np.save(path + 'Glo' + sensor_names[i] + '_ver2', x)

def exec_convert_world():
    main(training_path)
    main(val_path)
    main(test_path)
