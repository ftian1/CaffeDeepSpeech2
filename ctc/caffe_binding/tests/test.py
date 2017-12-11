"""
Tests are based on the torch7 bindings for warp-ctc. Reference numbers are also obtained from the tests.
"""
import unittest
import numpy as np
import ctc
from ctc import CTCLoss

ctc_loss = CTCLoss()
places = 5


def run_grads(label_sizes, labels, probs, sizes):
    cost = ctc_loss.ctc_loss(probs, labels, sizes, label_sizes)
    grads = ctc_loss.grads
    print "run_grads {}".format(type(grads))
    print "gradshape {}".format(grads.shape)
    print(grads.reshape(grads.shape[0] * grads.shape[1], grads.shape[2]))
    return cost


class TestCases(unittest.TestCase):
    def test_medium(self):
        probs = np.ascontiguousarray(np.array([
            [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.5, 0.2, 0.1]]
        ]), dtype=np.float32)

        labels = np.array([1, 2, 1, 2], dtype=np.int)
        label_sizes = np.array([2, 2], dtype=np.int)
        sizes = np.array([2, 2], dtype=np.int)
        cpu_cost = run_grads(label_sizes, labels, probs, sizes)
        expected_cost = 6.0165174007416
        #self.assertEqual(cpu_cost, gpu_cost)
        self.assertAlmostEqual(cpu_cost, expected_cost, places)

if __name__ == '__main__':
    unittest.main()
