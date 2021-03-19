"""Tests specific to the luetkenhaus scenario.
"""

import unittest
from scenarios.luetkenhaus.luetkenhaus import run
from libs.aux_functions import standard_bipartite_evaluation
import libs.matrix as mat
import numpy as np
import pandas as pd

class TestReproducabilityLuetkenhaus(unittest.TestCase):
    def test_run_seq(self):
        # simple reproducability test with fixed random seed
        np.random.seed(578164)
        length = 22e3
        p = run(L_TOT=length, max_iter=20, mode="seq")
        df = p.data
        fidelity_list = np.real_if_close([np.dot(np.dot(mat.H(mat.phiplus), state), mat.phiplus)[0, 0] for state in df["state"]])
        evaluation = standard_bipartite_evaluation(data_frame=df)
        self.assertTrue(np.allclose(list(df["time"]), [0.161223, 0.295175, 0.450631, 0.474823, 0.547847, 0.745639, 0.8613350000000001, 0.9955110000000001, 1.333079, 1.470727, 1.528071, 1.5327749999999998, 1.6632549999999997, 1.760247, 1.8401029999999998, 2.0601830000000003, 2.113719, 2.214071, 2.315991, 2.3686309999999997]))
        self.assertTrue(np.allclose(fidelity_list, [(0.9501471109778241+0j), (0.9129822690919149+0j), (0.90068538451139+0j), (0.9548808752993154+0j), (0.9504612129196545+0j), (0.9327892683658803+0j), (0.9516147276031116+0j), (0.9324366859935398+0j), (0.836589625127455+0j), (0.9416428842063612+0j), (0.9324870379814999+0j), (0.9566285793444832+0j), (0.9163188940840636+0j), (0.9239582977698734+0j), (0.951142491669352+0j), (0.9308283963844197+0j), (0.9460829357424178+0j), (0.9132713878665936+0j), (0.9333943387724394+0j), (0.9477459957104526+0j)]))
        self.assertEqual(list(df["resource_cost_max"]), [1287, 892, 1151, 154, 506, 1276, 909, 701, 2638, 913, 496, 29, 823, 667, 580, 1436, 248, 886, 478, 272])
        self.assertTrue(np.allclose(evaluation, [0.9308044199711022, 0.026620653523634588, 3.8245127488945108, 0.8679230633788223, 0.000554329914143113, 0.0001257979117325935]))

    def test_run_sim(self):
        # simple reproducability test with fixed random seed
        np.random.seed(578164)
        length = 22e3
        p = run(L_TOT=length, max_iter=20, mode="sim")
        df = p.data
        fidelity_list = np.real_if_close([np.dot(np.dot(mat.H(mat.phiplus), state), mat.phiplus)[0, 0] for state in df["state"]])
        evaluation = standard_bipartite_evaluation(data_frame=df)
        self.assertTrue(np.allclose(list(df["time"]), [0.144199, 0.244103, 0.373015, 0.39026299999999997, 0.44693499999999997, 0.589847, 0.691655, 0.770167, 1.065623, 1.167879, 1.223431, 1.2266789999999999, 1.3188549999999999, 1.393559, 1.458519, 1.619351, 1.647127, 1.746359, 1.799895, 1.830359]))
        self.assertTrue(np.allclose(fidelity_list, [(0.9014347440976141+0j), (0.9278781812783324+0j), (0.9119238276079461+0j), (0.9532976490837457+0j), (0.939387713959726+0j), (0.9181187479029569+0j), (0.9181674961920454+0j), (0.9474337177816496+0j), (0.8518026788453077+0j), (0.9274298597283939+0j), (0.9332934372230093+0j), (0.9573188443200803+0j), (0.9332429949233915+0j), (0.9338991856880656+0j), (0.9349612064685325+0j), (0.9122603215934882+0j), (0.9572125842815808+0j), (0.9137536844513777+0j), (0.9557274402750476+0j), (0.9542469463849256+0j)]))
        self.assertEqual(list(df["resource_cost_max"]), [1287, 892, 1151, 154, 506, 1276, 909, 701, 2638, 913, 496, 29, 823, 667, 580, 1436, 248, 886, 478, 272])
        self.assertTrue(np.allclose(evaluation, [0.929139563104361, 0.02433756309076234, 4.879345327841289, 1.0162826527016642, 0.0005465030984532036, 0.00011382707746398026]))

if __name__ == '__main__':
    unittest.main()
