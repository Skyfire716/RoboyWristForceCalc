# RoboyWristForceCalc

Libs needed:
Numpy 
Math
sympy
matplotlib
quaternion 

Install with (or other python package manager):
pip install numpy
pip install python-math
pip install sympy
pip install matplotlib
pip install numpy-quaternion

In the forcecalc.py file you can change the parameters that the mechanism has.
Use python3 forcecalc.py to run the script.
The script will create a EXPERIMENT*.txt file in which the force and torque vectors of the joint points at all tested positions are logged.

python3 getMax.py will read the Experiment file and print the 4 longest vectors (Probably the highest forces that occure).

How to use the forcecalc.py:
In line 37 you can configure the mechanisms parameters like alpha,l1-3 angles and the sphere radius.
In line 60 the pitch and yaw moveable area can be configured and the step size in which the area should be tested.

If the mechanism setup is wrong some configurations of the arms may be unreachable. (the script might crash if sympy cant find a valid position for the ul and ur position)


Consult the paper

K. Ueda, H. Yamada, H. Ishida, and S. Hirose, “Design of Large Motion Range and Heavy Duty 2-DOF Spherical Parallel Wrist Mechanism,” J. Robot. Mechatron., Vol.25 No.2, pp. 294-305, 2013.
https://doi.org/10.20965/jrm.2013.p0294
for more details about the mechanism and the configurations of the joints and angles.
