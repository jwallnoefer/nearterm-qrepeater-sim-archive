"""Test that resources are tracked properly for all processes that should do so.
"""
import unittest
from world import World
from quantum_objects import Pair, Station
from events import EntanglementSwappingEvent, DiscardQubitEvent, EntanglementPurificationEvent
import libs.matrix as mat
import numpy as np

_rho_phiplus = np.dot(mat.phiplus, mat.H(mat.phiplus))

def _epp_fail(rho):
    p_suc = 0
    new_state = np.dot(mat.phiplus, mat.H(mat.phiplus))
    return p_suc, new_state

def _epp_success(rho):
    p_suc = 1
    new_state = np.dot(mat.phiplus, mat.H(mat.phiplus))
    return p_suc, new_state

class TestResourceTracking(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.event_queue = self.world.event_queue

    def test_entanglement_swapping(self):
        stations = [Station(world=self.world, id=i, position=i * 200) for i in range(3)]
        qubits1 = [stations[0].create_qubit(), stations[1].create_qubit()]
        resource1 = np.random.random()*40
        pair1 = Pair(world=self.world, qubits=qubits1, initial_state=_rho_phiplus, initial_cost_add=resource1, initial_cost_max=resource1)
        qubits2 = [stations[1].create_qubit(), stations[2].create_qubit()]
        resource2 = np.random.random()*40
        pair2 = Pair(world=self.world, qubits=qubits2, initial_state=_rho_phiplus, initial_cost_add=resource2, initial_cost_max=resource2)
        # now schedule and resolve the event
        event = EntanglementSwappingEvent(time=0, pairs=[pair1, pair2], error_func=None)
        self.event_queue.add_event(event)
        self.event_queue.resolve_next_event()
        self.assertEqual(len(self.world.world_objects["Pair"]), 1)
        new_pair = self.world.world_objects["Pair"][0]
        self.assertEqual(new_pair.resource_cost_add, resource1 + resource2)
        self.assertEqual(new_pair.resource_cost_max, max(resource1, resource2))

    def test_discard_tracking(self):
        stations = [Station(world=self.world, id=i, position=i * 200) for i in range(2)]
        qubits1 = [stations[0].create_qubit(), stations[1].create_qubit()]
        resource1_add = np.random.random()*40
        resource1_max = np.random.random()*40
        pair1 = Pair(world=self.world, qubits=qubits1, initial_state=_rho_phiplus, initial_cost_add=resource1_add, initial_cost_max=resource1_max)
        event = DiscardQubitEvent(time=1, qubit=qubits1[0])
        self.event_queue.add_event(event)
        self.event_queue.resolve_next_event()
        qubits2 = [stations[0].create_qubit(), stations[1].create_qubit()]
        resource2_add = np.random.random()*40
        resource2_max = np.random.random()*40
        pair2 = Pair(world=self.world, qubits=qubits2, initial_state=_rho_phiplus, initial_cost_add=resource2_add, initial_cost_max=resource2_max)
        self.assertEqual(pair2.resource_cost_add, resource1_add + resource2_add)
        self.assertEqual(pair2.resource_cost_max, resource1_max + resource2_max)

    def test_epp_tracking_success(self):
        station1 = Station(world=self.world, id=1, position=0)
        station2 = Station(world=self.world, id=2, position=10)
        resource1_add = np.random.random()*40
        resource1_max = np.random.random()*40
        resource2_add = np.random.random()*40
        resource2_max = np.random.random()*40
        pair1 = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource1_add, initial_cost_max=resource1_max)
        pair2 = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource2_add, initial_cost_max=resource2_max)
        event = EntanglementPurificationEvent(time=self.world.event_queue.current_time, pairs=[pair1, pair2], protocol=_epp_success)
        self.world.event_queue.add_event(event)
        self.world.event_queue.resolve_next_event()
        self.assertEqual(len(self.world.world_objects["Pair"]), 1)
        pair = self.world.world_objects["Pair"][0]
        self.assertEqual(pair.resource_cost_add, resource1_add + resource2_add)
        self.assertEqual(pair.resource_cost_max, resource1_max + resource2_max)

    def test_epp_tracking_fail(self):
        station1 = Station(world=self.world, id=1, position=0)
        station2 = Station(world=self.world, id=2, position=10)
        resource1_add = np.random.random()*40
        resource1_max = np.random.random()*40
        resource2_add = np.random.random()*40
        resource2_max = np.random.random()*40
        pair1 = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource1_add, initial_cost_max=resource1_max)
        pair2 = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource2_add, initial_cost_max=resource2_max)
        event = EntanglementPurificationEvent(time=self.world.event_queue.current_time, pairs=[pair1, pair2], protocol=_epp_fail)
        self.world.event_queue.add_event(event)
        self.world.event_queue.resolve_next_event()
        self.assertEqual(len(self.world.world_objects["Pair"]), 0)
        resource3_add = np.random.random()*40
        resource3_max = np.random.random()*40
        pair = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource3_add, initial_cost_max=resource3_max)
        self.assertEqual(pair.resource_cost_add, resource1_add + resource2_add + resource3_add)
        self.assertEqual(pair.resource_cost_max, resource1_max + resource2_max + resource3_max)

    def test_tracked_pair_destruction(self):
        station1 = Station(world=self.world, id=1, position=0)
        station2 = Station(world=self.world, id=2, position=10)
        resource1_add = np.random.random()*40
        resource1_max = np.random.random()*40
        resource2_add = np.random.random()*40
        resource2_max = np.random.random()*40
        pair1 = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource1_add, initial_cost_max=resource1_max)
        pair1.destroy_and_track_resources()
        pair1.qubits[0].destroy()
        pair1.qubits[1].destroy()
        pair2 = Pair(world=self.world, qubits=[station1.create_qubit(), station2.create_qubit()], initial_state=_rho_phiplus, initial_cost_add=resource2_add, initial_cost_max=resource2_max)
        self.assertEqual(pair2.resource_cost_add, resource1_add + resource2_add)
        self.assertEqual(pair2.resource_cost_max, resource1_max + resource2_max)

if __name__ == '__main__':
    unittest.main()
