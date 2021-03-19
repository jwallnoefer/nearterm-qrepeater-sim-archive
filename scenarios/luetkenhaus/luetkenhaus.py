import os, sys; sys.path.insert(0, os.path.abspath("."))
from quantum_objects import Source, SchedulingSource, Station, Pair
from protocol import TwoLinkProtocol
from world import World
from events import SourceEvent, GenericEvent, EntanglementSwappingEvent
import libs.matrix as mat
import numpy as np
from libs.aux_functions import apply_single_qubit_map, x_noise_channel, y_noise_channel, z_noise_channel, w_noise_channel, assert_dir
import matplotlib.pyplot as plt
from warnings import warn

ETA_P = 0.66  # preparation efficiency
T_P = 2 * 10**-6  # preparation time
ETA_C = 0.04 * 0.3  # phton-fiber coupling efficiency * wavelength conversion
T_2 = 1  # dephasing time
C = 2 * 10**8 # speed of light in optical fiber
L_ATT = 22 * 10**3  # attenuation length
E_M_A = 0.01  # misalignment error
P_D = 10**-8  # dark count probability per detector
ETA_D = 0.3  # detector efficiency
P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 0.97  # BSM ideality parameter
F = 1.16  # error correction inefficiency

ETA_TOT = ETA_P * ETA_C * ETA_D


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t/dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return dephasing_noise_channel

def luetkenhaus_time_distribution(source):
    comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
    comm_time = 2 * comm_distance / C
    eta = ETA_TOT * np.exp(-comm_distance / L_ATT)
    eta_effective = 1 - (1 - eta) * (1 - P_D)**2
    trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
    random_num = np.random.geometric(eta_effective)
    return random_num * trial_time, random_num

def luetkenhaus_state_generation(source):
    state = np.dot(mat.phiplus, mat.H(mat.phiplus))
    # TODO needs more sophisticated handling for other scenarios - especially if not only the central station is faulty
    comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
    storage_time = 2 * comm_distance / C
    for idx, station in enumerate(source.target_stations):
        if station.memory_noise is not None:  # only central station has noisy storage
            state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
        if station.memory_noise is None:  # only count misalignment and dark counts for end stations
            # misalignment
            state = apply_single_qubit_map(map_func=y_noise_channel, qubit_index=idx, rho=state, epsilon=E_M_A)
            eta = ETA_TOT * np.exp(-comm_distance / L_ATT)
            # dark counts are modeled as white noise
            state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta))
    return state

def imperfect_bsm_err_func(four_qubit_state):
    return LAMBDA_BSM * four_qubit_state + (1-LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

def alpha_of_eta(eta):
    return eta * (1 - P_D) / (1 - (1 - eta) * (1 - P_D)**2)


class LuetkenhausProtocol(TwoLinkProtocol):
    """The Luetkenhaus Protocol.

    Parameters
    ----------
    world : World
        The world in which the protocol will be performed.
    mode : {"seq", "sim"}
        Selects sequential or simultaneous generation of links.

    Attributes
    ----------
    mode : str
        "seq" or "sim"

    """
    def __init__(self, world, mode="seq"):
        if mode != "seq" and mode != "sim":
            raise ValueError("LuetkenhausProtocol does not support mode %s. Use \"seq\" for sequential state generation, or \"sim\" for simultaneous state generation.")
        self.mode = mode
        super(LuetkenhausProtocol, self).__init__(world)

    def check(self):
        """Checks world state and schedules new events.

        Summary of the Protocol:
        Establish a left link and a right link.
        Then perform entanglement swapping.
        Record metrics about the long distance pair.
        Repeat.

        Returns
        -------
        None

        """
        # this protocol will only ever act if the event_queue is empty
        if self.world.event_queue:
            return
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        # if there are no pairs, begin protocol
        if not pairs:
            if self.mode == "seq":
                self.source_A.schedule_event()
            elif self.mode == "sim":
                self.source_A.schedule_event()
                self.source_B.schedule_event()
            return
        # in sequential mode, if there is only a pair on the left side, schedule creation of right pair
        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        if num_right_pairs == 0 and num_left_pairs == 1:
            if self.mode == "seq":
                self.source_B.schedule_event()
            return
        if num_right_pairs == 1 and num_left_pairs == 1:
            ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pairs[0], right_pairs[0]], error_func=imperfect_bsm_err_func)
            self.world.event_queue.add_event(ent_swap_event)
            return

        long_range_pairs = self._get_long_range_pairs()
        if long_range_pairs:
            long_range_pair = long_range_pairs[0]
            self._eval_pair(long_range_pair)
            # cleanup
            long_range_pair.qubits[0].destroy()
            long_range_pair.qubits[1].destroy()
            long_range_pair.destroy()
            self.check()
            return
        warn("LuetkenhausProtocol encountered unknown world state. May be trapped in an infinite loop?")


def run(L_TOT, max_iter, mode="seq"):
    world = World()
    station_A = Station(world, position=0, memory_noise=None)
    station_central = Station(world, position=L_TOT / 2, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_2))
    station_B = Station(world, position=L_TOT, memory_noise=None)
    source_A = SchedulingSource(world, position=L_TOT / 2, target_stations=[station_A, station_central], time_distribution=luetkenhaus_time_distribution, state_generation=luetkenhaus_state_generation)
    source_B = SchedulingSource(world, position=L_TOT / 2, target_stations=[station_central, station_B], time_distribution=luetkenhaus_time_distribution, state_generation=luetkenhaus_state_generation)
    protocol = LuetkenhausProtocol(world, mode=mode)
    protocol.setup()

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    p = run(L_TOT=22e3, max_iter=100, mode="seq")
    print(p.data)
