import numpy as np
import matplotlib.pyplot as plt

class Ellipse:
    def __init__(self, A=None, center=None, a=None, b=None, theta=0):
        """
        Initialize ellipse either from:
        - A (3x3 matrix for conic), OR
        - center=(h,k), semimajor a, semiminor b, rotation theta (radians)
          from the formula: (x-h)^2/a^2+(y-k)^2/b^2=1

        The conic in matrix form: [[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]]
        corresponds to Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """
        if A is not None:
            self.from_matrix(A)
        elif center is not None and a is not None and b is not None:
            self.from_parameters(center, a, b, theta)
        else:
            raise ValueError("Provide either matrix A or geometric parameters.")

    def from_parameters(self, center, a, b, theta):
        """Construct ellipse from geometric parameters."""
        self.h, self.k = center
        self.a = a
        self.b = b
        self.theta = theta

        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Diagonal form
        D = np.diag([1/a**2, 1/b**2])

        # Quadratic form matrix
        Q = R @ D @ R.T

        # Build full 3x3 conic matrix
        A = np.zeros((3, 3))
        A[:2, :2] = Q
        A[:2, 2] = -Q @ np.array([self.h, self.k])
        A[2, :2] = A[:2, 2]
        A[2, 2] = np.array([self.h, self.k]) @ Q @ np.array([self.h, self.k]) - 1

        self.A = A

    # def from_matrix(self, A):
    #     """Construct ellipse from 3x3 conic matrix."""
    #     self.A = A
    #
    #     # Extract components
    #     Q = A[:2, :2]
    #     b = A[:2, 2]
    #
    #     # Center
    #     center = -np.linalg.inv(Q) @ b
    #     self.h, self.k = center
    #
    #     # Translate to center
    #     T = np.eye(3)
    #     T[0, 2] = -self.h
    #     T[1, 2] = -self.k
    #
    #     A_c = T.T @ A @ T
    #     Q_c = A_c[:2, :2]
    #     F_c = A_c[2, 2]
    #
    #     # Eigen decomposition
    #     eigvals, eigvecs = np.linalg.eigh(Q_c)
    #
    #     # include F_c scaling
    #     if F_c >= 0:
    #         print("Not a real ellipse (F >= 0 after centering).")
    #
    #     axes = np.sqrt(-F_c / eigvals)
    #
    #     # Sort so that a >= b
    #     order = np.argsort(axes)[::-1]
    #     axes = axes[order]
    #     eigvecs = eigvecs[:, order]
    #
    #     self.a, self.b = axes
    #
    #     # Rotation angle (major axis direction)
    #     self.theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    def from_matrix(self, A):
        """Construct ellipse from 3x3 conic matrix."""
        self.A = A

        # Extract components
        Q = A[:2, :2]
        b = A[:2, 2]

        # Center = -Q^{-1} b
        center = -np.linalg.inv(Q) @ b
        self.h, self.k = center

        # Translate to center
        T = np.eye(3)
        T[0, 2] = -self.h
        T[1, 2] = -self.k

        A_c = T.T @ A @ T
        Q_c = A_c[:2, :2]

        # Eigen decomposition to get axes
        eigvals, eigvecs = np.linalg.eigh(Q_c)

        self.a = 1 / np.sqrt(eigvals[0])
        self.b = 1 / np.sqrt(eigvals[1])

        # Rotation angle
        self.theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    def parametric_points(self, num=200):
        """Generate points on the ellipse."""
        t = np.linspace(0, 2*np.pi, num)
        x = self.a * np.cos(t)
        y = self.b * np.sin(t)

        # Rotate and translate
        R = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta),  np.cos(self.theta)]
        ])

        pts = R @ np.vstack((x, y))
        pts[0] += self.h
        pts[1] += self.k

        return pts

    def plot(self, ax=None, lw=2, color='r', **kwargs):
        """Plot the ellipse using matplotlib."""
        if ax is None:
            fig, ax = plt.subplots()

        pts = self.parametric_points()
        ax.plot(pts[0], pts[1], lw=lw, color=color, **kwargs)

        # Draw center
        #ax.plot(self.h, self.k, 'ro')

        ax.set_aspect('equal')
        # center ax ranges on ellipse:
        #ax.set(xlim=[self.h-1.4*2*self.a,self.h+1.4*2*self.a],
        #       ylim=[self.k-1.4*2*self.b,self.k+1.4*2*self.b])
        return ax

    def is_valid_ellipse(self, tol=1e-10):
        """
        Check whether the stored 3x3 matrix represents a real ellipse.
        Returns True/False and diagnostic info.
        """
        A = self.A

        # Extract coefficients
        Axx = A[0, 0]
        Axy = 2 * A[0, 1]
        Ayy = A[1, 1]
        Dx = 2 * A[0, 2]
        Ey = 2 * A[1, 2]
        F  = A[2, 2]

        # 1. Discriminant test
        discriminant = Axy**2 - 4 * Axx * Ayy
        is_ellipse_type = discriminant < -tol

        # 2. Quadratic form matrix
        Q = A[:2, :2]

        # Check positive definiteness (eigenvalues > 0)
        eigvals = np.linalg.eigvals(Q)
        is_positive_definite = np.all(eigvals > tol)

        # 3. Check it's a real (non-empty) ellipse
        # Compute center
        try:
            center = -np.linalg.inv(Q) @ A[:2, 2]
            val = np.array([*center, 1.0]) @ A @ np.array([*center, 1.0])
            is_real = val < 0  # must be negative for real ellipse
        except np.linalg.LinAlgError:
            is_real = False

        valid = is_ellipse_type and is_positive_definite and is_real

        return {
            "valid": valid,
            "discriminant": discriminant,
            "is_ellipse_type": is_ellipse_type,
            "positive_definite": is_positive_definite,
            "is_real": is_real,
            "eigenvalues": eigvals
        }

# Example usage
if __name__ == "__main__":
    # From parameters
    e1 = Ellipse(center=(2, 1), a=5, b=3, theta=np.pi/6)
    ax = e1.plot(color='blue', label='Parametric')

    # From matrix
    A = e1.A
    e2 = Ellipse(A=A)
    e2.plot(ax=ax, linestyle='--', color='red', label='From params')

    ax.legend()
    plt.show()

    A = np.array([
        [5, 1, -7],
        [1, 3, -7],
        [-7, -7, 13]
        ])
    # Create ellipse from matrix
    e = Ellipse(A=A)
    # Print extracted geometric parameters
    print(f"Center: ({e.h:.3f}, {e.k:.3f})")
    print(f"Semi-axes: a={e.a:.3f}, b={e.b:.3f}")
    print(f"Rotation (radians): {e.theta:.3f}")

    # Plot
    ax = e.plot(color='blue', label='Ellipse from matrix')
    ax.legend()
    plt.show()
