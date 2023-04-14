import numpy as np
import math

file = open('EXPERIMENTForcePositionCheck.txt', 'r')
lines = file.readlines()

FwlList = []
FulList = []
FvlList = []
FwrList = []
FurList = []
FvrList = []
FdList = []
tauwlList = []
tauwrList = []
taudList = []

def getSorted(l):
    return sorted(l, key=lambda tup: math.fabs(tup[0]))

for line in lines:
    vars = line.split(',')
    #print(vars)
    vec = np.array([0, 0, 0], dtype=np.float64)
    count = 0
    for var in vars:
        if count % 3 == 0:
            if count == 3:
                FwlList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 6:
                FulList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 9:
                FvlList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 12:
                FwrList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 15:
                FurList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 18:
                FvrList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 21:
                FdList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 24:
                tauwlList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 27:
                tauwrList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
            if count == 30:
                taudList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))

        var = var.replace('\'', '').replace(')', '').replace('(', '').replace("F_wl=", '').replace("F_ul=", '').replace("F_vl=", '').replace("F_wr=", '').replace('F_ur=', '').replace("F_vr=", '').replace("F_d=", '').replace("twl=", '').replace("twr=", '').replace("td=", '')
        #print("Var Count ", var, count)
        if "pitch" not in var and "yaw" not in var:
            count += 1
            vec[count % 3] = float(var)
            #print(vec)
    #print(FwlList)
    taudList.append((math.sqrt(vec.dot(vec)), np.copy(vec)))
print('\n\n\n')
sortedFwl = getSorted(FwlList)
sortedFul = getSorted(FulList)
sortedFvl = getSorted(FvlList)
sortedFwr = getSorted(FwrList)
sortedFur = getSorted(FurList)
sortedFvr = getSorted(FvrList)
sortedFd = getSorted(FdList)
sortedtwl = getSorted(tauwlList)
sortedtwr = getSorted(tauwrList)
sortedtd = getSorted(taudList)

print("Fwl ", sortedFwl[-4:])
#print(sortedFwl)
print("Ful ", sortedFul[-4:])
print("Fvl ", sortedFvl[-4:])
print("Fwr ", sortedFwr[-4:])
print("Fur ", sortedFur[-4:])
print("Fvl ", sortedFvr[-4:])
print("Fd ", sortedFd[-4:])
print("twl ", sortedtwl[-4:])
#print(sortedtwl)
print("twr ", sortedtwr[-4:])
print("td ", sortedtd[-4:])
