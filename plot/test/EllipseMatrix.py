import numpy as np
import matplotlib.pyplot as plt

class EllipseMatrix:
    def __init__(self, matrix):
        self.M = np.array(matrix, dtype=float)
        if self.M.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")

        # Ensure symmetry
        self.M = (self.M + self.M.T) / 2  # Force symmetry
        self.is_valid, self.error_msg = self.validate()
        if not self.is_valid:
            print(f"Warning: {self.error_msg}")
        else:
            self._extract_properties()

    def _extract_properties(self):
        # Matrix form: [[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]]
        # Corresponding to Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        A = self.M[0, 0]
        B = 2 * self.M[0, 1]
        C = self.M[1, 1]
        D = 2 * self.M[0, 2]
        E = 2 * self.M[1, 2]
        F = self.M[2, 2]

        # 1. Find Center: Solve system [2A B; B 2C] * [x; y] = -[D; E]
        sub_m = np.array([[2*A, B], [B, 2*C]])
        self.center = np.linalg.solve(sub_m, -np.array([D, E]))

        # 2. Translate to origin to find axes (K is the adjusted constant term)
        # Equation becomes: Ax'^2 + Bx'y' + Cy'^2 + K = 0
        K = F - self.center.T @ (np.array([[A, B/2], [B/2, C]]) @ self.center + np.array([D, E]))

        # 3. Eigen-decomposition of the quadratic part A' = [[A, B/2], [B/2, C]]
        quad_part = self.M[:2, :2]
        eigenvals, eigenvecs = np.linalg.eigh(quad_part)

        # Semi-axis lengths: L = sqrt(-K / eigenvalue)
        # Note: If -K/eigenval < 0, it's not a real ellipse
        if (any(-K / eigenvals) < 0 or any(eigenvals) == 0):
            self.is_valid = False
        else:
            self.axes = np.sqrt(-K / eigenvals)
        self.angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    def plot(self, ax=None, lw=2, color='r', **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        x0,y0 = self.center
        a,b = self.axes
        # Generate points in parameter space
        t = np.linspace(0, 2*np.pi, 100)
        # Base ellipse at origin: (x,y)=(acos(t),bsin(t)); from (x-x0)^2/a^2+(y-y0)^2/b^2=1
        coords = np.array([a * np.cos(t), b * np.sin(t)])

        # Rotate and Translate
        theta = np.radians(self.angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        rotated_coords = R @ coords
        final_coords = rotated_coords + self.center[:, np.newaxis]

        ax.plot(final_coords[0], final_coords[1], lw=lw, color=color, **kwargs)
        ax.set_aspect('equal')
        #ax.set(xlim=[x0-1.4*2*a,x0+1.4*2*a],
        #       ylim=[y0-1.4*2*b,y0+1.4*2*b]) # center ax ranges on ellipse
        return ax

    def validate(self):
        # Extract components
        A = self.M[0, 0]
        B = 2 * self.M[0, 1]
        C = self.M[1, 1]

        # 1. Check if it's a degenerate conic (det == 0)
        if np.isclose(np.linalg.det(self.M), 0):
            return False, "Degenerate conic (point or lines)."

        # 2. Check type (B^2 - 4AC < 0 for ellipses)
        discriminant = B**2 - 4*A*C
        if discriminant >= 0:
            return False, f"Not an ellipse (Discriminant {discriminant} >= 0). Likely hyperbola/parabola."

        # 3. Check for imaginary ellipse
        # For a real ellipse, the trace (A+C) and det(M) must have opposite signs
        if np.linalg.det(self.M) * (A + C) > 0:
            return False, "Imaginary ellipse (no real solutions)."

        return True, "Valid real ellipse."


    @classmethod
    def from_params(cls, center, major, minor, angle_deg=0):
        """
        Creates an EllipseMatrix instance from geometric parameters.
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

def main():
    # Example:
    # Create an ellipse at (5, 5), axes 10 and 5, rotated 45 degrees
    newe = EllipseMatrix.from_params(center=(0,3), major=10, minor=5, angle_deg=45)
    newe.plot(color='red', label='My Ellipse')
    plt.show()

    printe=EllipseMatrix.from_params(center=(2,4), major=12, minor=7, angle_deg=45)
    print(printe.center, printe.axes, printe.angle)
    #printe.plot(color='blue', label='My printe')
    #plt.show()

    # Example usage:
    # A circle at (2,3) with radius 5: (x-2)^2 + (y-3)^2 = 25
    # Expands to: x^2 - 4x + 4 + y^2 - 6y + 9 - 25 = 0 => x^2 + y^2 - 4x - 6y - 12 = 0
    matrix = [[1, 0, -2],
          [0, 1, -3],
          [-2, -3, -12]]

    e = EllipseMatrix(matrix)
    print(f"Center: {e.center}, Axes: {e.axes}, Angle: {e.angle}°")
    e.plot(color='red', label='My Ellipse')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
