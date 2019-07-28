# ===================================================
# tb03rd tests
import unittest
from slycot import math
import numpy as np
import numpy.linalg as la
from numpy.testing import assert_raises, assert_almost_equal

np.set_printoptions(linewidth=150)
CASES = {}

# Unique eigenvalues, with large enough pmax, should be able to
# completely diagonalize.
CASES['case0'] = {
    'A': np.array([
        [-0.75, 1.0,  2.0,  3.0],
        [0.0, -1.0,  1.0,  0.0],
        [0.0,  0.0, -0.25,  1.0],
        [0.0,  0.0,  0.0, -1.50]]),
    'A_bdschur': np.array([
        [-0.750, 0.0,  0.0,  0.0],
        [0.0,   -1.0,  0.0,  0.0],
        [0.0,    0.0, -0.25, 0.0],
        [0.0,    0.0,  0.0, -1.50]]),
    'X0': np.eye(4),
    # Obtain this from eig(A)
    'X_exp': np.array(
        [[1.0, -0.97014,  0.9701, -0.88998832],
         [0.0,  0.24253,  0.19402,  0.35599533],
         [0.0,  0.00000,  0.14552, -0.17799766],
         [0.0,  0.00000,  0.00000,  0.22249708]]),
    'pmax': 1.0e3,
    'tol': 1.0e-2,
    'sort': 'N'
}

# Non-unique eigenvalues. We get something that looks
# like jordan blocks. Checked in matlab
CASES['case1'] = {
    'A': np.array([
        [-0.5, 1.0,  2.0,  3.0],
        [0.0, -1.0,  1.0,  0.0],
        [0.0,  0.0, -0.5,  1.0],
        [0.0,  0.0,  0.0, -1.0]]),
    'A_bdschur': np.array([
        [-0.50,  1.7889,  0.0000,  0.0000],
        [0.0,   -0.5000,  0.0000,  0.0000],
        [0.0,    0.0000, -1.0000,  0.2408],
        [0.0,    0.0000,  0.0000, -1.0000]]),
    'X0': np.eye(4),
    'X_exp': np.array([
        [1.0000, 0.0000,  0.8944,  0.9691],
        [0.0000, 0.8944, -0.4472, -0.2154],
        [0.0000, 0.4472,  0.0000, -0.1077],
        [0.0000, 0.0000,  0.0000,  0.0538]]),
    'pmax': 1.0e3,
    'tol': 1.0e-2,
    'sort': 'N'
}

# Example from MB03RD documentation.
CASES['slicot_example'] = {
    'A': np.array([
        [1.0,   -1.0,    1.0,    2.0,  3.0, 1.0,  2.0,    3.0],
        [1.0,    1.0,    3.0,    4.0,  2.0, 3.0,  4.0,    2.0],
        [0.0,    0.0,    1.0,   -1.0,  1.0, 5.0,  4.0,    1.0],
        [0.0,    0.0,    0.0,    1.0, -1.0, 3.0,  1.0,    2.0],
        [0.0,    0.0,    0.0,    1.0,  1.0, 2.0,  3.0,   -1.0],
        [0.0,    0.0,    0.0,    0.0,  0.0, 1.0,  5.0,    1.0],
        [0.0,    0.0,    0.0,    0.0,  0.0, 0.0,  0.99999999,   -0.99999999],
        [0.0,    0.0,    0.0,    0.0,  0.0, 0.0,  0.99999999,    0.99999999]]),
    'A_bdschur': np.array([
        [1.0, -1.0, -1.2247, -0.7071, -3.4186,  1.4577,  0.0000,  0.0000],
        [1.0,  1.0,  0.0000,  1.4142, -5.1390,  3.1637,  0.0000,  0.0000],
        [0.0,  0.0,  1.0000, -1.7321, -0.0016,  2.0701,  0.0000,  0.0000],
        [0.0,  0.0,  0.5774,  1.0000,  0.7516,  1.1379,  0.0000,  0.0000],
        [0.0,  0.0,  0.0000,  0.0000,  1.0000, -5.8606,  0.0000,  0.0000],
        [0.0,  0.0,  0.0000,  0.0000,  0.1706,  1.0000,  0.0000,  0.0000],
        [0.0,  0.0,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -0.8850],
        [0.0,  0.0,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000]]),
    'X0': np.eye(8),
    'X_exp': np.array([
        [1.0, 0.0,  0.0000,  0.0000,  0.0000,  0.0000,  0.9045,  0.1957],
        [0.0, 1.0,  0.0000,  0.0000,  0.0000,  0.0000, -0.3015,  0.9755],
        [0.0, 0.0,  0.8165,  0.0000, -0.5768, -0.0156, -0.3015,  0.0148],
        [0.0, 0.0, -0.4082,  0.7071, -0.5768, -0.0156,  0.0000, -0.0534],
        [0.0, 0.0, -0.4082, -0.7071, -0.5768, -0.0156,  0.0000,  0.0801],
        [0.0, 0.0,  0.0000,  0.0000, -0.0276,  0.9805,  0.0000,  0.0267],
        [0.0, 0.0,  0.0000,  0.0000,  0.0332, -0.0066,  0.0000,  0.0000],
        [0.0, 0.0,  0.0000,  0.0000,  0.0011,  0.1948,  0.0000,  0.0000]]),
    'pmax': 1.0e3,
    'tol': 1.0e-2,
    'sort': 'N'
}


class TestMb03rd(unittest.TestCase):

    def test_mb03rd(self):
        """
        Test that mb03rd computes the correct block diagonal form,
        and that the returned transformation X is correct.
        """
        for key in CASES:
            case = CASES[key]
            self.check_mb03rd(case)

    def check_mb03rd(self, case):
        """
        check case
        """
        n = case['A'].shape[0]
        A = case['A']
        X0 = case['X0']

        eigs_exp = la.eigvals(A)
        eigs_exp = eigs_exp + 0*1j*eigs_exp
        eigs_exp.sort()

        out = math.mb03rd(A, pmax=case['pmax'],  tol=case['tol'],
                          sort=case['sort'])

        A_bdschur = out[0]
        X_act = out[1]
        nblks = out[2]
        blsize = out[3]
        eigs_act = out[4]

        eigs_act.sort()
        assert_almost_equal(eigs_exp, eigs_act)
        assert_almost_equal(case['A_bdschur'], A_bdschur, decimal=4)
        assert_almost_equal(la.solve(X_act, A.dot(X_act)), A_bdschur)

    def test_mb03rd_errors(self):
        """
        Check the error handling of mb03md. We give wrong inputs and
        and check that this raises an error.
        """

        # test error handling
        n = 5
        A_og = np.random.randn(n, n)
        X_og = np.eye(n)
        # non-square A
        assert_raises(ValueError, math.mb03rd, A_og[:, 0:n-1])
        assert_raises(ValueError, math.mb03rd, A_og[0:n-1, :])

        # non-square X
        assert_raises(ValueError, math.mb03rd, A_og, X=X_og[0:n-1, :])
        assert_raises(ValueError, math.mb03rd, A_og, X=X_og[:, 0:n-1])

        # invalid sort
        assert_raises(ValueError, math.mb03rd, A_og, X=X_og, sort='P')

        # pmax >=1
        assert_raises(ValueError, math.mb03rd, A_og, X=X_og, pmax=-1.0)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestMb03rd)


if __name__ == "__main__":
    unittest.main()
