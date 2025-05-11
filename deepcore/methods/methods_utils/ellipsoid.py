import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import eigh

class EllipsoidND:
    """
    Class representing an ellipsoid in R^n.
    
    Attributes:
        G (np.ndarray): Shape (n, n), positive definite symmetric matrix defining the ellipsoid.
        c (np.ndarray): Shape (n,), center of the ellipsoid.
        n (int): Dimension of the space.
    """
    def __init__(self, G, c):
        """
        Initialize an ellipsoid E(G, c) = { x | (x-c)^T G (x-c) <= 1 }.
        
        Args:
            G (np.ndarray): Shape (n, n), positive definite symmetric matrix.
            c (np.ndarray): Shape (n,), center of the ellipsoid.
        """
        G = np.asarray(G)
        c = np.asarray(c)
        
        # Check dimensions
        if G.shape[0] != G.shape[1]:
            raise ValueError("G must be a square matrix.")
        if G.shape[0] != c.shape[0]:
            raise ValueError("Dimension of G and c must match.")
        
        # Check symmetry
        if not np.allclose(G, G.T):
            raise ValueError("G must be symmetric.")
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(G)
        if not np.all(eigenvalues > 0):
            raise ValueError("G must be positive definite.")
        
        self.G = G
        self.c = c
        self.n = G.shape[0]
        self.vertices = self.find_ellipsoid_vertices()
    
    def plot_ellipsoid(self, ax=None):
        """
        Plot the ellipsoid. Supports 2D and 3D visualization.
        
        Args:
            ax: Matplotlib axis object (optional).
        
        Returns:
            ax: Matplotlib axis object.
        """
        if self.n > 3:
            raise ValueError("Visualization is only supported for 2D or 3D.")
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d' if self.n == 3 else None)
        
        if self.n == 2:
            # Compute ellipse parameters
            eigenvalues, eigenvectors = eigh(self.G)
            theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
            width = 2 / np.sqrt(eigenvalues[0])
            height = 2 / np.sqrt(eigenvalues[1])
            
            # Create ellipse
            ellipse = Ellipse(xy=self.c, width=width, height=height, angle=theta,
                            edgecolor='red', fc='None', lw=2, label='ellipsoid')
            ax.add_patch(ellipse)
            ax.scatter(self.c[0], self.c[1], color='red', label='center')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
        
        elif self.n == 3:
            # Compute ellipsoid parameters
            eigenvalues, eigenvectors = eigh(self.G)
            # Radii of the ellipsoid
            radii = 1 / np.sqrt(eigenvalues)
            
            # Parametric equations for a unit sphere
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Transform to ellipsoid
            for i in range(len(u)):
                for j in range(len(v)):
                    point = np.array([x[i, j], y[i, j], z[i, j]])
                    point = eigenvectors @ (radii * point) + self.c
                    x[i, j], y[i, j], z[i, j] = point
            
            ax.plot_surface(x, y, z, color='red', alpha=0.3)
            ax.scatter(self.c[0], self.c[1], self.c[2], color='red', label='center')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        
        return ax
    
    def shrink_ellipsoid(self, r):
        """
        Shrink the ellipsoid E(G, c) around its center c by a factor of 1/r.
        
        Args:
            r (float): Scaling factor (r > 0).
        
        Returns:
            EllipsoidND: New ellipsoid with scaled matrix.
        """
        if r <= 0:
            raise ValueError("Scaling factor r must be positive.")
        G_prime = r**2 * self.G
        return EllipsoidND(G_prime, self.c)
    
    def find_ellipsoid_vertices(self):
        """
        Return the 2n vertices of the ellipsoid E(G, c), corresponding to the endpoints
        of the principal axes.
        
        Returns:
            np.ndarray: Array of shape (2n, n), each row is a vertex.
        """
        # Use eigh for symmetric matrices
        eigenvalues, eigenvectors = eigh(self.G)
        # Compute semi-axes lengths
        semi_axes = 1 / np.sqrt(eigenvalues)
        
        vertices = []
        for i in range(self.n):
            # Vertices along the i-th principal axis
            v_plus = self.c + semi_axes[i] * eigenvectors[:, i]
            v_minus = self.c - semi_axes[i] * eigenvectors[:, i]
            vertices.extend([v_plus, v_minus])
        
        return np.array(vertices)

# Example usage
if __name__ == "__main__":
    # 2D example
    G_2d = np.array([[4, 1], [1, 2]])
    c_2d = np.array([1, 1])
    ellipsoid_2d = EllipsoidND(G_2d, c_2d)
    
    fig, ax = plt.subplots()
    ellipsoid_2d.plot_ellipsoid(ax)
    shrunk_2d = ellipsoid_2d.shrink_ellipsoid(2)
    shrunk_2d.plot_ellipsoid(ax)
    plt.title("2D Ellipsoid and Shrunk Ellipsoid")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('ellipsoid_2d.png')
    
    # 3D example
    G_3d = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    c_3d = np.array([0, 0, 0])
    ellipsoid_3d = EllipsoidND(G_3d, c_3d)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ellipsoid_3d.plot_ellipsoid(ax)
    shrunk_3d = ellipsoid_3d.shrink_ellipsoid(2)
    shrunk_3d.plot_ellipsoid(ax)
    plt.title("3D Ellipsoid and Shrunk Ellipsoid")
    plt.savefig('ellipsoid_3d.png')