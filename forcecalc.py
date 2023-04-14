import numpy as np
import math
import sympy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import quaternion as quat

def skew(x):
    return np.matrix([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]], dtype=np.float64)

def zeroos():
    return np.repeat([np.zeros((3,3))], 12, axis=0)

def vec(v):
    return "({:.2f},{:.2f},{:.2f})".format(v[0].item(), v[1].item(), v[2].item())

def angle(u, v):
    return math.degrees(math.acos(np.dot(u, v) / (math.sqrt(u.dot(u)) * math.sqrt(v.dot(v)))))

def rot(v, axis, angle):
    axis_angle = (math.radians(angle)*0.5) * axis/np.linalg.norm(axis)
    vec = quat.quaternion(*v)
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)
    v_prime = q * vec * np.conjugate(q)
    res = v_prime.imag
    return np.array([res[0], res[1], res[2]])

E3 = np.identity(3)


np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})


#Change the parameters of the mechanism here
#SphereRadius in meters
sphereRadius = 0.04
#Angles in degree
alpha = 37.5
l1 = 45
l2 = 87
l3 = 40



#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlim([-sphereRadius, sphereRadius])
#ax.set_ylim([-sphereRadius, sphereRadius])
#ax.set_zlim([-sphereRadius, sphereRadius])
#plt.ion()
#plt.show()            
            
with open('EXPERIMENTForcePositionCheck.txt', 'w') as f:
    
    #Pitch and yaw min max values measured from the center
    for pitch in np.linspace(-70, 80, num=75, endpoint=False):
        for yaw in np.linspace(-40, 50, num=45, endpoint=False):
            #yaw = 20
            #pitch = 40
            print("Pitch ", pitch)
            print("Yaw ", yaw)
            thau_p = pitch
            F = np.array([0, 0, 0])
            #Regular Moment
            #tau = np.array([0.3, 1.5, 0.3])
            #Moment with reduced Sphereradius while full Wrist Plate Length
            #Moment in NewtonMeter 
            tau = np.array([0.3, 2.0, 0.3])

            #1.5kg * 9,81m/s^2 ~= 15N
            # 15N * 0.1m = 1.5Nm

            #print("F ", F)
            #print("tau ", tau)
            Ry = np.matrix([[math.cos(thau_p), 0, math.sin(thau_p)],
                        [0, 1, 0],
                        [-math.sin(thau_p), 0, math.cos(thau_p)]])



            #d = np.array([math.cos(math.radians(yaw)) + math.sin(math.radians(pitch)), math.sin(math.radians(yaw)), math.sin(math.radians(pitch))])
            d = np.array([1, 0, 0])
            d = rot(d, np.array([0, 1, 0]), pitch)
            d = rot(d, np.array([0, 0, 1]), yaw)
            zm = np.array([0, 0, 1])
            zm = rot(zm, np.array([0, 1, 0]), pitch)

            #further improve rotation
            #check that zm ym d are all 90deg
            ym = np.cross(d, zm)

            
            #print("D,zm ", angle(d, zm))
            #print("D, ym ", angle(d, ym))
            #print("zm,ym ", angle(ym, zm))

            test = zm
            test = rot(test, ym, -l3)
            vl = rot(test, zm, alpha)
            vr = rot(test, zm, -alpha)
            #test = rot(test, ym, l3)

            d = sphereRadius * d
            zm = sphereRadius * zm
            test = sphereRadius * test

            wl = np.array([0, sphereRadius, 0])
            #vl = np.array([math.sin(math.radians(l3 + pitch)) * math.cos(math.radians(alpha + yaw)), math.sin(math.radians(l3 + pitch)) * math.sin(math.radians( alpha + yaw)), math.cos(math.radians(l3 + pitch))])
            #print("Alpha + yaw ", alpha + yaw)
            #print("VL x ", vl[0])
            #print("VL y ", vl[1])
            #print("VL z ", vl[2])
            vl = sphereRadius * vl
            ul1, ul2, ul3 = sympy.symbols("ul1 ul2 ul3", real=True)
            eq1 = sympy.Eq(math.cos(math.radians(l2)) * sphereRadius ** 2, vl[0] * ul1 + vl[1] * ul2 + vl[2] * ul3)
            eq2 = sympy.Eq(math.cos(math.radians(l1)) * sphereRadius ** 2, wl[0] * ul1 + wl[1] * ul2 + wl[2] * ul3)
            eq3 = sympy.Eq(sphereRadius**2, ul1**2+ul2**2+ul3**2)
            #result = sympy.solve([eq1, eq2, eq3])
            (ulr1, ulr2) = sympy.nonlinsolve([eq1, eq2, eq3], [ul1, ul2, ul3])
            #print("Ul ", result)
            #ul = np.array([result[0][ul1], result[0][ul2], result[0][ul3]])
            ul = np.array([ulr1[0], ulr1[1], ulr1[2]])
            
            #print("vlul ", angle(vl, ul))
            #print("ulwl ", angle(ul, wl))

            #print("Sphere ", sphereRadius)
            #print("Ul Lenght ", math.sqrt(ul.dot(ul)))
            if not (0.95 <= (math.sqrt(ul.dot(ul)) / sphereRadius) <= 1.05):
                print("Sphere ", sphereRadius)
                print("Ul Lenght ", math.sqrt(ul.dot(ul)))
                exit()
            
            if not (0.95 <= (math.sqrt(vl.dot(vl)) / sphereRadius) <= 1.05):
                print("Spphere ", sphereRadius)
                print("Vl Length ", math.sqrt(vl.dot(vl)))
                exit()

            wr = np.array([0, -sphereRadius, 0])
            #vr = np.array([math.sin(math.radians(l3 + pitch)) * math.cos(math.radians(-alpha + yaw)), math.sin(math.radians(l3 + pitch)) * math.sin(math.radians(-alpha + yaw)), math.cos(math.radians(l3 + pitch))])

            #print("alpha + yaw ", -alpha + yaw)
            #print("VR x ", vr[0])
            #print("VR y ", vr[1])
            #print("VR z ", vr[2])
            vr = sphereRadius * vr

            ur1, ur2, ur3 = sympy.symbols("ur1 ur2 ur3", real=True)
            equr1 = sympy.Eq(math.cos(math.radians(l2)) * sphereRadius ** 2, vr[0] * ur1 + vr[1] * ur2 + vr[2] * ur3)
            equr2 = sympy.Eq(math.cos(math.radians(l1)) * sphereRadius ** 2, wr[0] * ur1 + wr[1] * ur2 + wr[2] * ur3)
            equr3 = sympy.Eq(sphereRadius**2, ur1**2+ur2**2+ur3**2)
            #resultUR = sympy.solve([equr1, equr2, equr3])
            (urr1, urr2) = sympy.nonlinsolve([equr1, equr2, equr3], [ur1, ur2, ur3])
            #ur = np.array([resultUR[0][ur1], resultUR[0][ur2], resultUR[0][ur3]])
            ur = np.array([urr1[0], urr1[1], urr1[2]])
            
            #print("vr ur ", angle(vr, ur))
            #print("ur, wr", angle(wr, ur))

            #print("Angle Between VL VR ", math.degrees(math.acos(vl.dot(vr) / (math.sqrt(vl.dot(vl)) * math.sqrt(vr.dot(vr))))))

            #if not (0.95 <= math.degrees(vl.dot(vr) / (math.sqrt(vl.dot(vl)) * math.sqrt(vr.dot(vr)))) / 2 * alpha <= 1.05):
                #print("Alpha ", alpha)
                #print("VL VR ", math.degrees(math.acos(vl.dot(vr) / (math.sqrt(vl.dot(vl)) * math.sqrt(vr.dot(vr))))))
                #print("Should be double")
                #exit()

            #test = np.array([math.sin(math.radians(l3 + pitch)) * math.cos(math.radians(0)), 0, math.cos(math.radians(l3 + pitch))])
            #test = sphereRadius * test

            #print("Test VL ", (math.degrees(math.acos(vl.dot(test) /(math.sqrt(vl.dot(vl)) * math.sqrt(test.dot(test)))))))

            soa = np.array([[0, 0, 0, wl[0], wl[1], wl[2]], [0, 0, 0, vl[0], vl[1], vl[2]], [0, 0, 0, ul[0], ul[1], ul[2]], [0, 0, 0, wr[0], wr[1], wr[2]], [0, 0, 0, vr[0], vr[1], vr[2]], [0, 0, 0, ur[0], ur[1], ur[2]], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])

            X, Y, Z, U, V, W  = zip(*soa)
            
            #ax.quiver(X, Y, Z, U, V, W, color=['r', 'r', 'r', 'r', 'r', 'r'])


            #ax.quiver(0, 0, 0, 0, sphereRadius, 0, color='r')
            #ax.quiver(0, 0, 0, sphereRadius, 0, 0, color='g')
            #ax.quiver(0, 0, 0, 0, 0, sphereRadius, color='b')
            
            #ax.quiver(0,0,0,d[0],d[1],d[2], color='g')
            #ax.quiver(0,0,0,zm[0],zm[1],zm[2], color='b')
            #ax.quiver(0,0,0,ym[0],ym[1],ym[2], color='r')
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = sphereRadius * np.cos(u)*np.sin(v)
            y = sphereRadius * np.sin(u)*np.sin(v)
            z = sphereRadius * np.cos(v)
            #ax.plot_wireframe(x, y, z, color="r")

            #plt.draw()
            #plt.pause(0.001)

            if not (0.95 <= (math.sqrt(ur.dot(ur)) / sphereRadius) <= 1.05):
                print("Sphere ", sphereRadius)
                print("Ur length ", math.sqrt(ur.dot(ur)))
                exit()
                
            if not (0.95 <= (math.sqrt(vr.dot(vr)) / sphereRadius) <= 1.05):
                print("Sphere ", sphereRadius)
                print("Vr Lenght ", math.sqrt(vr.dot(vr)))
                exit()

            skewVL = skew(vl)
            skewVR = skew(vr)
            skewULWL = skew(ul - wl)
            skewVLUL = skew(vl - ul)
            skewURWR = skew(ur - wr)
            skewVRUR = skew(vr - ur)
            
            A = np.matrix([[1,0,0, -1,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,1,0, 0,-1,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,1, 0,0,-1, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],

                        [0,0,0, -skewULWL.item(0),-skewULWL.item(1),-skewULWL.item(2), 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0],
                        [0,0,0, -skewULWL.item(3),-skewULWL.item(4),-skewULWL.item(5), 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0],
                        [0,0,0, -skewULWL.item(6),-skewULWL.item(7),-skewULWL.item(8), 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0],
                        
                        [0,0,0, 1,0,0, -1,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,1,0, 0,-1,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,1, 0,0,-1, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, -skewVLUL.item(0),-skewVLUL.item(1),-skewVLUL.item(2), 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, -skewVLUL.item(3),-skewVLUL.item(4),-skewVLUL.item(5), 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, -skewVLUL.item(6),-skewVLUL.item(7),-skewVLUL.item(8), 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0,    1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0,    0,1,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0,    0,0,1, 0,0,1, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, skewVL.item(0),skewVL.item(1),skewVL.item(2), 0,0,0, 0,0,0,    skewVR.item(0),skewVR.item(1),skewVR.item(2), 0,0,0, 0,0,0, 0,0,0, 1,0,0],
                        [0,0,0, 0,0,0, skewVL.item(3),skewVL.item(4),skewVL.item(5), 0,0,0, 0,0,0,    skewVR.item(3),skewVR.item(4),skewVR.item(5), 0,0,0, 0,0,0, 0,0,0, 0,1,0],
                        [0,0,0, 0,0,0, skewVL.item(6),skewVL.item(7),skewVL.item(8), 0,0,0, 0,0,0,    skewVR.item(6),skewVR.item(7),skewVR.item(8), 0,0,0, 0,0,0, 0,0,0, 0,0,1],
                        
                        [0,0,0, 0,0,0, 0,0,0, 1,0,0, -1,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,-1,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,1, 0,0,-1,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, -skewURWR.item(0),-skewURWR.item(1),-skewURWR.item(2),    0,0,0, 0,0,0, 0,0,0, 1,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, -skewURWR.item(3),-skewURWR.item(4),-skewURWR.item(5),    0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, -skewURWR.item(6),-skewURWR.item(7),-skewURWR.item(8),    0,0,0, 0,0,0, 0,0,0, 0,0,1, 0,0,0],
                        
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0,    -1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0,    0,-1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1,    0,0,-1, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    -skewVRUR.item(0),-skewVRUR.item(1),-skewVRUR.item(2), 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    -skewVRUR.item(3),-skewVRUR.item(4),-skewVRUR.item(5), 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    -skewVRUR.item(6),-skewVRUR.item(7),-skewVRUR.item(8), 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                        
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, math.cos(thau_p), 0, math.sin(thau_p)],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0],
                        [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,    0,0,0, 0,0,0, 0,0,0, 0,0,0, -math.sin(thau_p), 0, math.cos(thau_p)]], dtype=np.float64)

            b = np.array([0.0,0,0, 0,0,0, 0,0,0, 0,0,0, -F[0],-F[1],-F[2], -tau[0],-tau[1],-tau[2], 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0], dtype=np.float64)
            b = np.transpose([b])

            forceResult = np.linalg.pinv(A) * b
            #print(A)
            #print("A Shape ", A.shape)
            #print(b)
            #print("b shape ", b.shape)
            #forceResult = np.linalg.lstsq(A, b, rcond=None)[0]

            #print("ForceResult ", forceResult)

            Fwl = np.transpose(np.array([forceResult[0], forceResult[1], forceResult[2]]))
            Ful = np.transpose(np.array([forceResult[3], forceResult[4], forceResult[5]]))
            Fvl = np.transpose(np.array([forceResult[6], forceResult[7], forceResult[8]]))
            Fwr = np.transpose(np.array([forceResult[9], forceResult[10], forceResult[11]]))
            Fur = np.transpose(np.array([forceResult[12], forceResult[13], forceResult[14]]))
            Fvr = np.transpose(np.array([forceResult[15], forceResult[16], forceResult[17]]))
            Fd = np.transpose(np.array([forceResult[18], forceResult[19], forceResult[20]]))
            twl = np.transpose(np.array([forceResult[21], forceResult[22], forceResult[23]]))
            twr = np.transpose(np.array([forceResult[24], forceResult[26], forceResult[26]]))
            td = np.transpose(np.array([forceResult[27], forceResult[28], forceResult[29]]))

            Fwl = np.array([forceResult[0], forceResult[1], forceResult[2]])
            Ful = np.array([forceResult[3], forceResult[4], forceResult[5]])
            Fvl = np.array([forceResult[6], forceResult[7], forceResult[8]])
            Fwr = np.array([forceResult[9], forceResult[10], forceResult[11]])
            Fur = np.array([forceResult[12], forceResult[13], forceResult[14]])
            Fvr = np.array([forceResult[15], forceResult[16], forceResult[17]])
            Fd = np.array([forceResult[18], forceResult[19], forceResult[20]])
            twl = np.array([forceResult[21], forceResult[22], forceResult[23]])
            twr = np.array([forceResult[24], forceResult[26], forceResult[26]])
            td = np.array([forceResult[27], forceResult[28], forceResult[29]])

            
            f.write("pitch={:.2f},yaw={:.2f},F_wl={},F_ul={},F_vl={},F_wr={},F_ur={},F_vr={},F_d={},twl={},twr={},td={}\n".format(pitch, yaw, vec(Fwl), vec(Ful), vec(Fvl), vec(Fwr), vec(Fur), vec(Fvr), vec(Fd), vec(twl), vec(twr), vec(td)))
            

            #print("Done")
print("Simulation Done")
