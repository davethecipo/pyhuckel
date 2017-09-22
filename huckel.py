import os.path

import numpy as np
from numpy.linalg import eigvalsh, norm
from scipy.linalg import eigvals, fractional_matrix_power


class HuckelOverlap(object):
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

        Raise an Exception if the norm of the imaginary vector of eigenvalues
        is bigger that energy_tolerance."""
        hamiltonian = self._hamiltonian(theta1, theta2)
        overlap = self._overlap(theta1, theta2)
        overlap_inverse12 = fractional_matrix_power(overlap, -0.5)
        right = np.dot(hamiltonian, overlap_inverse12)
        Hp = np.dot(overlap_inverse12.T.conj(), right)
        En = eigvals(Hp)
        self._check_energy_imaginary_component(En, energy_tolerance)
        return [np.sort(En.real)]

    @staticmethod
    def _check_energy_imaginary_component(energy_array, tolerance=10e-10):
        """Raise Exception if the norm of the imaginary part of energy_array
        is greater than tolerance and exit(1)"""
        residue = norm(energy_array.imag)
        if residue > tolerance:
            raise Exception('The energy eigenvalues have an imaginary part that\
            is too big. Aborting.')

    def _hamiltonian(self, theta1, theta2):
        return self.cyclic(self.a, self.b1, self.b2, theta1, theta2)

    def _overlap(self, theta1, theta2):
        return self.cyclic(self.s0, self.s1, self.s2, theta1, theta2)

    @staticmethod
    def cyclic(matrix, matrix1, matrix2, theta1, theta2):
        """Return a cyclic matrix that satisfies the Born Von Karman
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


class Huckel(HuckelOverlap):
    """Run Huckel calculation without overlap."""
    def __init__(self, a, b1, b2):
        problem_size = a.shape
        s0 = np.ones(problem_size)
        s1 = np.zeros(problem_size)
        # s1 and s2 are both 0 matrices
        super().__init__(a, b1, b2, s0, s1, s1)

    def energy_eigenvalues(self, theta1, theta2, energy_tolerance=10e-10):
        """Calculate  energy eigenvalues for given phase factors theta1 and theta2.

        Raise an Exception if the norm of the imaginary vector of eigenvalues
        is bigger that energy_tolerance."""
        hamiltonian = self._hamiltonian(theta1, theta2)
        En = eigvalsh(hamiltonian)
        self._check_energy_imaginary_component(En, energy_tolerance)
        return [np.sort(En.real)]


class Path(object):
    def __init__(self, start, end, Npoints):
        self.start = start
        self.end = end
        self.Npoints = Npoints

    def __iter__(self):
        """Return a list with length self.Npoints of points following the
        path from self.start to self.end"""
        for idx in range(0, self.Npoints):
            position = self.start + (self.end-self.start)/self.Npoints*idx
            yield position
        raise StopIteration()


class CalcRunner(object):
    def __init__(self, huckel):
        self.problem = huckel

    def save_band_for_path(self, path, filename):
        """Save the energy eigenvalues of huckel for each point along path in the
         reciprocal space to filename"""
        with open(filename, 'ab') as file_hander:
            for point in path:
                energies = self.problem.energy_eigenvalues(point[0], point[1])
                np.savetxt(file_hander, energies)

    @staticmethod
    def is_there_a_previous_run(filename):
        previous = False
        if os.path.exists(filename):
            print('File {} already exists!'.format(filename))
            previous = True
        return previous
