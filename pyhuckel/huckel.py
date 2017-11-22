import numpy as np
from numpy.linalg import eigvalsh, norm
from scipy.linalg import fractional_matrix_power


class EnergyError(Exception):
    def __init__(self, energy_eigenvalues, msg=None):
        if msg is None:
            msg = "Error with the Hamiltonian eigenvalues: complex values are" \
                  "present. Array of energy values: {}".format(energy_eigenvalues)
        super(EnergyError, self).__init__(msg)
        self.energy_eigenvalues = energy_eigenvalues


class Huckel(object):
    """Run Huckel calculation without overlap."""
    def __init__(self, a, b1, b2):
        self.a = a
        self.b1 = b1
        self.b2 = b2

    def energy_eigenvalues(self, theta1, theta2, energy_tolerance=10e-10):
        """Calculate  energy eigenvalues for given phase factors theta1 and theta2.

        Raise EnergyError if the norm of the imaginary vector of eigenvalues
        is bigger that energy_tolerance."""
        hamiltonian = self._hamiltonian(theta1, theta2)
        En = eigvalsh(hamiltonian)
        self._check_energy_imaginary_component(En, energy_tolerance)
        return [np.sort(En.real)]

    @staticmethod
    def _check_energy_imaginary_component(energy_array, tolerance=10e-10):
        """Raise EnergyError if the norm of the imaginary part of energy_array
        is greater than tolerance"""
        residue = norm(energy_array.imag)
        if residue > tolerance:
            raise EnergyError(energy_array)

    def _hamiltonian(self, theta1, theta2):
        return self.circulant(self.a, self.b1, self.b2, theta1, theta2)

    @staticmethod
    def circulant(matrix, matrix1, matrix2, theta1, theta2):
        """Return a circulant matrix that satisfies the Born Von Karman
        boundary conditions.

        Args:
            matrix: describes the interactions in the unit cell
            matrix1: describes the interactions of the unit cell with the
             neighbour in direction 1
            matrix2: describes the interactions of the unit cell with the
             neighbour in direction 2
            theta1: phase factor along direction 1
            theta2: phase factor along direction 2
        """
        phase1 = np.exp(1j*theta1)
        phase2 = np.exp(1j*theta2)
        return matrix + matrix1*phase1 + matrix1.T.conj()*phase1.conjugate() + \
            matrix2*phase2 + matrix2.T.conj()*phase2.conjugate()


class HuckelOverlap(Huckel):
    """Allow Huckel type calculation including overlap"""
    def __init__(self, a, b1, b2, s0, s1, s2):
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2

    def energy_eigenvalues(self, theta1, theta2, energy_tolerance=10e-10):
        """Calculate  energy eigenvalues for given phase factors theta1 and theta2.

        Raise EnergyError if the norm of the imaginary vector of eigenvalues
        is bigger that energy_tolerance."""
        hamiltonian = self._hamiltonian(theta1, theta2)
        overlap = self._overlap(theta1, theta2)
        overlap_inverse12 = fractional_matrix_power(overlap, -0.5)
        right = np.dot(hamiltonian, overlap_inverse12)
        Hp = np.dot(overlap_inverse12.T.conj(), right)
        En = eigvalsh(Hp)
        self._check_energy_imaginary_component(En, energy_tolerance)
        return [np.sort(En.real)]

    def _overlap(self, theta1, theta2):
        return self.circulant(self.s0, self.s1, self.s2, theta1, theta2)
