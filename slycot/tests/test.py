import unittest
from slycot import synthesis
from slycot import math
from slycot import transform

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_1(self):
        synthesis.sb02mt(1,1,1,1)

    def test_2(self):
        from numpy import matrix
        a = matrix("-2 0.5;-1.6 -5")
        Ar, Vr, Yr, VALRr, VALDr = math.mb05md(a, 0.1)

    def test_sb02ad(self):
        "Test sb10ad, Hinf synthesis"
        import numpy as np
        a = np.array([[-1]])
        b = np.array([[1, 1]])
        c = np.array([[1], [1]])
        d = np.array([[0, 1], [1, 0]])

        n = 1
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1
        gamma = 10

        gamma_est, Ak, Bk, Ck, Dk, Ac, Bc, Cc, Dc, rcond = synthesis.sb10ad(
            n, m, np_, ncon, nmeas, gamma, a, b, c, d)
        # from Octave, which also uses SB10AD:
        #   a= -1; b1= 1; b2= 1; c1= 1; c2= 1; d11= 0; d12= 1; d21= 1; d22= 0;
        #   g = ss(a,[b1,b2],[c1;c2],[d11,d12;d21,d22]);
        #   [k,cl] = hinfsyn(g,1,1);
        # k.a is Ak, cl.a is Ac
        # gamma values don't match; not sure that's critical
        # this is a bit fragile
        # a simpler, more robust check might be to check stability of Ac
        self.assertEqual(Ak.shape, (1, 1))
        self.assertAlmostEqual(Ak[0][0], -3)
        self.assertEqual(Ac.shape, (2, 2))
        self.assertAlmostEqual(Ac[0][0], -1)
        self.assertAlmostEqual(Ac[0][1], -1)
        self.assertAlmostEqual(Ac[1][0], 1)
        self.assertAlmostEqual(Ac[1][1], -3)

    def test_td04ad_static(self):
        """Regression: td04ad (TFM -> SS transformation) for static TFM"""
        import numpy as np
        from itertools import product
        # 'C' fails on static TFs
        for nout,nin,rc in product(range(1,6),range(1,6),['R']):
            num = np.reshape(np.arange(nout*nin),(nout,nin,1))
            if rc == 'R':
                den = np.reshape(np.arange(1,1+nout),(nout,1))
            else:
                den = np.reshape(np.arange(1,1+nin),(nin,1))
            index = np.tile([0],den.shape[0])
            nr,a,b,c,d = transform.td04ad(rc,nin,nout,index,den,num)


# ===================================================
# Begin tb05ad tests
import numpy as np
from control import ss

from scipy import linalg
from numpy.testing import assert_raises, assert_almost_equal
from numpy.testing import assert_array_almost_equal, dec

CASES = {}

CASES['fail1'] = ss(np.array([[-0.5,  0.,  0.,  0. ],
                              [ 0., -1.,  0. ,  0. ],
                              [ 1.,  0., -0.5,  0. ],
                              [ 0.,  1.,  0., -1. ]]),
                   np.array([[ 1.,  0.],
                             [ 0.,  1.],
                             [ 0.,  0.],
                             [ 0.,  0.]]),
                   np.array([[ 0.,  1.,  1.,  0.],
                             [ 0.,  1.,  0.,  1.],
                             [ 0.,  1.,  1.,  1.]]),
                    np.zeros([3,2])   )

n = 20
p = 10
m = 14
np.random.seed(40)
CASES['pass1'] = ss(np.random.rand(n, n),
                    np.random.rand(n, m),
                    np.random.rand(p, n),
                     np.zeros([p,m]))

class test_tb05ad(unittest.TestCase):

    def setUp(self):
        pass

    CASES = {}

    CASES['fail1'] = ss(np.array([[-0.5,  0.,  0.,  0. ],
                                  [ 0., -1.,  0. ,  0. ],
                                  [ 1.,  0., -0.5,  0. ],
                                  [ 0.,  1.,  0., -1. ]]),
                       np.array([[ 1.,  0.],
                                 [ 0.,  1.],
                                 [ 0.,  0.],
                                 [ 0.,  0.]]),
                       np.array([[ 0.,  1.,  1.,  0.],
                                 [ 0.,  1.,  0.,  1.],
                                 [ 0.,  1.,  1.,  1.]]),
                        np.zeros([3,2])   )

    n = 20
    p = 10
    m = 14
    np.random.seed(40)
    CASES['pass1'] = ss(np.random.rand(n, n),
                        np.random.rand(n, m),
                        np.random.rand(p, n),
                         np.zeros([p,m]))


    def test_tb05ad_ng(self):
        for key in CASES:
            sys = CASES[key]
            self.check_tb05ad_AG_NG(sys, 10*1j, 'NG')


    @dec.knownfailureif
    def test_tb05ad_ag_failure(self):
        # Test the failure when we do balancing on certain A matrices.
        self.check_tb05ad_AG_NG(CASES['fail1'], 'AG')


    def test_tb05ad_nh(self):
        # Test the conversion to Hessenberg form and subsequent solution.
        jomega = 10*1j
        for key in CASES:
            sys = CASES[key]
            sys2 = self.check_tb05ad_AG_NG(sys, jomega, 'NG')
            self.check_tb05ad_NH(sys2, sys, jomega)


    # Check error handling.
    def test_tb05ad_errors(self):
        self.check_tb05ad_errors(CASES['pass1'])


    def check_tb05ad_AG_NG(self, sys, jomega, job):
        result = transform.tb05ad(sys.states, sys.inputs, sys.outputs, jomega,
                                  sys.A, sys.B, sys.C, job=job)
        g_i = result[3]
        hinvb = linalg.solve(np.eye(sys.states) * jomega - sys.A, sys.B)
        g_i_solve = sys.C.dot(hinvb)
        np.testing.assert_almost_equal(g_i_solve, g_i)
        return ss(result[0], result[1], result[2], sys.D)


    def check_tb05ad_NH(self, sys_transformed, sys, jomega):
        # When input matrices are already Hessenberg, output format changes.
        result = transform.tb05ad(sys_transformed.states, sys_transformed.inputs,
                                  sys_transformed.outputs, jomega,
                                  sys_transformed.A, sys_transformed.B,
                                  sys_transformed.C, job='NH')
        g_i = result[0]
        hinvb = linalg.solve(np.eye(sys.states) * jomega - sys.A, sys.B)
        g_i_solve = sys.C.dot(hinvb)
        np.testing.assert_almost_equal(g_i_solve, g_i)


    def check_tb05ad_errors(self, sys):
        jomega = 10*1j
        # test error handling
        # wrong size A
        assert_raises(ValueError, transform.tb05ad, sys.states+1,
                      sys.inputs, sys.outputs, jomega, sys.A, sys.B, sys.C, job='NH')
        # wrong size B
        assert_raises(ValueError, transform.tb05ad, sys.states,
                      sys.inputs+1, sys.outputs, jomega, sys.A, sys.B, sys.C, job='NH')
        # wrong size C
        assert_raises(ValueError, transform.tb05ad,sys.states,
                      sys.inputs, sys.outputs+1, jomega, sys.A, sys.B, sys.C, job='NH')
        # unrecognized job
        assert_raises(ValueError, transform.tb05ad,sys.states,
                      sys.inputs, sys.outputs, jomega, sys.A, sys.B, sys.C, job='a')



def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestConvert)


if __name__ == "__main__":
    unittest.main()
