import logging
import os.path

import numpy as np

logger = logging.getLogger(__name__)

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
            logger.info('File {} already exists.'.format(filename))
            previous = True
        return previous
