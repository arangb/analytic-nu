import numpy as np
import pylab as plt
from ellipsetools import Ellipse as ET
import EllipseMatrix as EM
from gpt_ellipse import Ellipse as GPT

e1=np.array([[1,0,0],[0,4,0],[0,0,-4]])
e2=np.array([[1,3,0],[3,2,0],[1,0,-1]])
e3=np.array([[1,0,-1],[0,2,0],[-1,0,-1]])
e4=np.array([[1,0,-1],[0,1/9,2/9],[-1,2/9,1]]) # (x-1)^2+(y+2)^2/9=1
e5=np.array([[5, 1, -7],[1, 3, -7],[-7, -7, 13]]) # center should be (1,2)


def printEM(e):
    print(e.center,e.axes,e.angle)

def printET(e):
    print(e.center_of_ellipse,e.axes_semimaj_semimin,e.angle_of_ellipse)

def printGPT(e):
    print("Center =",e.h,e.k," Axes = ",e.a,e.b," Angle = ", e.theta)

for i,e in enumerate([e1,e2,e3,e4,e5]):
    print('========================================')
    print(e)
    fig, ax = plt.subplots()

    test_em = EM.EllipseMatrix(e)
    if (test_em.is_valid):
        print("EM")
        printEM(test_em)
        test_em.plot(ax=ax, color='red', label='EllipseMatrix', linestyle='--')

    test_et = ET(e)
    if (test_et):
        print("ET")
        printET(test_et)
        test_et.plot_ellipse(ax=ax, edgecolor='blue', label='EllipseTools', linestyle='-.')

    test_gpt = GPT(e)
    if (test_gpt.is_valid_ellipse):
        print("GPT")
        printGPT(test_gpt)
        test_gpt.plot(ax=ax, color='black', label='GPT',linestyle=':')

    plt.legend()
    plt.savefig("ellipse_test_"+str(i)+".png")


eMpar=EM.EllipseMatrix.from_params(center=(2,4), major=12, minor=7, angle_deg=45)
eTpar=ET.from_params(center=(2,4), major=12, minor=7, angle_deg=45)
GPTpar=GPT(center=(2,4), a=12, b=7, theta=np.radians(45))
print("EMpar")
printEM(eMpar)
print("ETpar")
printET(eTpar)
print("GPTpar")
printGPT(GPTpar)

fig, ax = plt.subplots()
eMpar.plot(ax=ax, color='red', label='EllipseMatrix', linestyle='--')
eTpar.plot_ellipse(ax=ax, edgecolor='blue', label='EllipseTools', linestyle='-.')
GPTpar.plot(ax=ax, color='black', label='GPT',linestyle=':')
plt.legend()
plt.savefig("ellipse_test_param.png")
