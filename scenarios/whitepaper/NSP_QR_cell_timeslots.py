import os, sys; sys.path.insert(0, os.path.abspath("."))
from quantum_objects import SchedulingSource, Station
from protocol import TwoLinkProtocol
from world import World
from events import EntanglementSwappingEvent
import libs.matrix as mat
import numpy as np
from libs.aux_functions import apply_single_qubit_map, y_noise_channel, z_noise_channel, w_noise_channel
from noise import NoiseModel, NoiseChannel
from warnings import warn
from functools import lru_cache

C = 2 * 10**8  # speed of light in optical fiber
L_ATT = 22 * 10**3  # attenuation length


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return dephasing_noise_channel


def construct_y_noise_channel(epsilon):
    return lambda rho: y_noise_channel(rho=rho, epsilon=epsilon)


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d)**2)


class NSPProtocol(TwoLinkProtocol):
    """Short summary.

    Parameters
    ----------
    world : World
        The world in which the protocol will be performed.
    mode : {"seq", "sim"}
        Selects sequential or simultaneous generation of links.

    Attributes
    ----------
    time_list : list of scalars
    fidelity_list : list of scalars
    correlations_z_list : list of scalars
    correlations_x_list : list of scalars
    resource_cost_max_list : list of scalars
    mode : str
        "seq" or "sim"

    """

    def __init__(self, world, communication_speed, mode="seq"):
        if mode != "seq" and mode != "sim":
            raise ValueError("NSPProtocol does not support mode %s. Use \"seq\" for sequential state generation, or \"sim\" for simultaneous state generation.")
        self.mode = mode
        super(NSPProtocol, self).__init__(world=world, communication_speed=communication_speed)

    def check(self):
        """Checks world state and schedules new events.

        Summary of the Protocol:
        Establish a left link and a right link.
        Then perform entanglement swapping.
        Record metrics about the long distance pair.
        Repeat.
        However, sometimes pairs will be discarded because of memory cutoff
        times, which means we need to check regularly if that happened.

        Returns
        -------
        None

        """
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        num_left_pairs_scheduled = len(self._left_pairs_scheduled())
        num_right_pairs_scheduled = len(self._right_pairs_scheduled())
        # schedule pair creation if there is no pair and pair creation is not already scheduled
        if self.mode == "sim":
            if num_left_pairs == 0 and num_left_pairs_scheduled == 0:
                self.source_A.schedule_event()
            if num_right_pairs == 0 and num_right_pairs_scheduled == 0:
                self.source_B.schedule_event()
        elif self.mode == "seq":
            if not pairs and num_left_pairs_scheduled == 0 and num_right_pairs_scheduled == 0:
                self.source_A.schedule_event()
            elif num_left_pairs == 1 and num_right_pairs == 0 and num_right_pairs_scheduled == 0:
                self.source_B.schedule_event()
            elif num_right_pairs == 1 and num_left_pairs == 0 and num_left_pairs_scheduled == 0:  # this might happen if left pair is discarded
                self.source_A.schedule_event()
        # rest continues the same for both modes
        if num_left_pairs == 1 and num_right_pairs == 1:
            ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pairs[0], right_pairs[0]])
            # print("an entswap event was scheduled at %.8f while event_queue looked like this:" % self.world.event_queue.current_time, self.world.event_queue.queue)
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
        if not self.world.event_queue.queue:
            warn("Protocol may be stuck in a state without events.")


def run(length, max_iter, params, m=None, mode="sim"):
    # unpack the parameters
    try:
        P_LINK = params["P_LINK"]
    except KeyError:
        P_LINK = 1.0
    try:
        T_P = params["T_P"]  # preparation time
    except KeyError:
        T_P = 0
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError:
        T_DP = 1.0
    try:
        E_MA = params["E_MA"]  # misalignment error
    except KeyError:
        E_MA = 0
    try:
        P_D = params["P_D"]  # dark count probability
    except KeyError:
        P_D = 0
    try:
        LAMBDA_BSM = params["LAMBDA_BSM"]
    except KeyError:
        LAMBDA_BSM = 1

    T_0 = (length / 2) / C
    cutoff_time = m
    T_DP = T_DP / T_0  # everything time related needs to be in units of T_0

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def time_distribution(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = 2
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time, random_num

    @lru_cache()  # CAREFUL: only makes sense if positions and errors do not change!
    def state_generation(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        storage_time = 2
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # only central station has noisy storage
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
            if station.dark_count_probability is not None:  # dark counts are handled here because the information about eta is needed for that
                eta = P_LINK * np.exp(-comm_distance / L_ATT)
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability))
        return state

    misalignment_noise = NoiseChannel(n_qubits=1, channel_function=construct_y_noise_channel(epsilon=E_MA))
    world = World()
    station_A = Station(world, position=0, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    station_central = Station(world, position=length / 2,
                              memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP),
                              memory_cutoff_time=cutoff_time,
                              BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                              )
    station_B = Station(world, position=length, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    source_A = SchedulingSource(world, position=length / 2, target_stations=[station_A, station_central], time_distribution=time_distribution, state_generation=state_generation)
    source_B = SchedulingSource(world, position=length / 2, target_stations=[station_central, station_B], time_distribution=time_distribution, state_generation=state_generation)
    protocol = NSPProtocol(world, communication_speed=C * T_0, mode=mode)
    protocol.setup()

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    p = run(length=22000, max_iter=100, params={"P_LINK": 5 * 10**-2,
                           "f_clock": 5 * 10**6,
                           "T_DP": 100 * 10**-3}, m=2, mode="sim")
    print(p.data)
