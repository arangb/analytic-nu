import numpy as np
import matplotlib as plt

'''
An Ellipse class to define ellipses in 3x3 matrix form:

Follow wikipedia notation: Ax^2+Bxy+Cy^2+Dx+Ey+F=0.
Amatrix=[[A  , B/2, D/2],
         [B/2, C  , E/2],
         [D/2, E/2, F]]

https://en.wikipedia.org/wiki/Ellipse#General_ellipse

and plot a given ellipse. You can also define the ellipse in canonical form (A-F).

# https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot-python
# https://math.stackexchange.com/questions/2870880/generalization-of-ellipse-in-matrix-representation
# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
#https://math.stackexchange.com/questions/3076317/what-is-the-equation-for-an-ellipse-in-standard-form-after-an-arbitrary-matrix-t
# https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=0CAMQw7AJahcKEwiAvoaUjej8AhUAAAAAHQAAAAAQAg&url=https%3A%2F%2Fwww.geometrictools.com%2FDocumentation%2FInformationAboutEllipses.pdf&psig=AOvVaw0f-DnytSEMs-wmXGKroojj&ust=1674919833667218
#https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjrwZyGo-_8AhUxgf0HHWfgBToQFnoECBAQAQ&url=http%3A%2F%2Fwww.cs.cornell.edu%2Fcourses%2Fcs422%2F2008sp%2FA6%2FEllipse.pdf&usg=AOvVaw1SfA0RoqMM6GtnGNE74q0E

AGB: May 2026: I tried creating similar classes with ChatGPT and this one is in the end more robust. It doesn't fail in many cases like when using matrices for the transformations'
'''

class Ellipse(object):
    ''' Example: e=Ellipse(np.matrix([[1,-2,-3],[-2,5,-6],[-3,-6,9]]))'''

    def __init__(self, matrix):
        if matrix.size != 9:
            print("Matrix should be 3x3")

        A = matrix[0, 0]
        B = 2*matrix[0, 1]
        C = matrix[1, 1]
        D = 2*matrix[0, 2]
        E = 2*matrix[1, 2]
        F = matrix[2, 2]

        if B**2-4*A*C > 1.e-6:
            print("This is not an ellipse: B**2-4*A*C  is positive?")

        for item in ['A', 'B', 'C', 'D', 'E', 'F']:
            setattr(self, item, eval(item))

    @staticmethod
    def canonical_to_matrix(A, B, C, D, E, F):
        '''Ax^2+Bxy+Cy^2+Dx+Ey+F=0. '''
        Amatrix = np.matrix([[A, B/2, D/2],
                             [B/2, C, E/2],
                             [D/2, E/2, F]])
        return Amatrix

    @classmethod
    def canonical(cls, A, B, C, D, E, F):
        '''You can use this as a second constructor: e=Ellipse.canonical(a,b,c,d,e,f)'''
        return cls(*canonical_to_matrix(A, B, C, D, E, F))

    def matrix_to_canonical(self):
        return (self.A, self.B, self.C, self.D, self.E, self.F)

    def check_ellipse(self):
        # 1. Discriminant < 1
        check = self.B**2-4*self.A*self.C < 1.e-6
        if not check:
            print("ellipsetools: this is not an ellipse B^2-4AC>1!!")
        return check

    @classmethod
    def from_params(cls, center, major, minor, angle_deg=0):
        """
        Creates an ellipse instance from geometric parameters.
        center: (h, k)
        major, minor: semi-axis lengths
        angle_deg: rotation in degrees
        """
        h, k = center
        a, b = major, minor
        theta = np.radians(angle_deg)

        # 1. Canonical matrix (centered at origin, no rotation)
        # x^2/a^2 + y^2/b^2 - 1 = 0
        M_canonical = np.diag([1/(a**2), 1/(b**2), -1])

        # 2. Rotation matrix (in homogeneous coordinates)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos_t,  sin_t, 0],
            [-sin_t, cos_t, 0],
            [0,      0,     1]
        ])

        # 3. Translation matrix
        T = np.array([
            [1, 0, -h],
            [0, 1, -k],
            [0, 0,  1]
        ])

        # 4. Transform: M = T.T @ R.T @ M_canonical @ R @ T
        # (This moves the point from world space back to canonical space)
        M_final = T.T @ R.T @ M_canonical @ R @ T

        return cls(M_final)

    @property
    def axes_semimaj_semimin(self):
        ''' Return the axis major and minor from the A matrix. 
        Then the cartesian equation of the ellipse is given by xT.M.x=1
        with M=(A)^{-T}A^{-1} and the axes are 1/sqrt{lambda_i} 
        where lambda_i are the eigenvalues of M. And the eigenvectors are
        aligned with the axes.'''
        # M = np.linalg.inv(A.T).dot(np.linalg.inv(A))
        # eigenvals,_ = np.linalg.eig(M)
        # semimaj,semimin = sorted(eigenvals)

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        F = self.F
        semimaj, seminin = 0., 0.
        bra1 = 2*(A*E**2+C*D**2-B*D*E+(B**2-4*A*C)*F)
        bra2 = np.sqrt((A-C)**2+B**2)

        semimaj = -np.sqrt(bra1*((A+C)+bra2))/(B**2-4*A*C)
        semimin = -np.sqrt(bra1*((A+C)-bra2))/(B**2-4*A*C)
        if semimaj < semimin:
            print("Warning! axes maj and min are swapped??")

        return semimaj, semimin

    @property
    def center_of_ellipse(self):
        ''' Return the center of an ellipse in matrix representation (x-x0)^2/a+(y-y0)^2/b=1'''
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        F = self.F
        x0 = (2*C*D-B*E)/(B**2-4*A*C)
        y0 = (2*A*E-B*D)/(B**2-4*A*C)
        return x0, y0

    @property
    def angle_of_ellipse(self):
        ''' https://math.stackexchange.com/questions/2733847/how-can-i-calculate-the-angle-of-an-ellipse-given-its-matrix-representation

        This is the formula from wikipedia: "angle" is the angle from the positive horizontal axis to the ellipse's major axis.
        '''

        A = self.A
        B = self.B
        C = self.C
        angle = np.arctan((C-A-np.sqrt((A-C)**2+B**2))/B)
        if abs(B) < 1.e-6 and A < C:
            angle = 0.
        elif abs(B) < 1.e-6 and A > C:
            angle = np.pi/2
        return angle

    @property
    def eccentrity(self):
        a, b = self.axes_semimaj_semimin
        return np.sqrt(1-(b/a)**2)

    def plot_ellipse(self, ax, lw=2, edgecolor='r', facecolor='none', **kwargs):
        if not self.check_ellipse():
            print("This is not a well-defined ellipse, cannot plot it")
            return
        x0, y0 = self.center_of_ellipse
        a, b = self.axes_semimaj_semimin
        theta = self.angle_of_ellipse  # in rads
        # the width is 2*semimajor axis, and the height is 2*semiminor axis.
        ell = plt.patches.Ellipse((x0, y0), width=2*a, height=2*b, angle=np.degrees(
            theta), lw=lw, ec=edgecolor, fc=facecolor, **kwargs)
        # ax.add_patch(ell)
        # ax.set(xlim=[x0-1.2*2*a,x0+1.2*2*a],
        #       ylim=[y0-1.2*2*b,y0+1.2*2*b]) # center ax ranges on ellipse
        return ax.add_patch(ell)
