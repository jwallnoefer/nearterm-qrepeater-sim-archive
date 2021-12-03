import os, sys; sys.path.insert(0, os.path.abspath("."))
from quantum_objects import SchedulingSource, Source, Station
from protocol import Protocol, MessageReadingProtocol
from world import World
from events import SourceEvent, EntanglementSwappingEvent, EntanglementPurificationEvent
import libs.matrix as mat
import numpy as np
from libs.aux_functions import apply_single_qubit_map, y_noise_channel, z_noise_channel, w_noise_channel
from warnings import warn
from collections import defaultdict
from noise import NoiseModel, NoiseChannel

# WARNING: this protocol will not work with memory time outs as that is managed separately by the protocol


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


def construct_w_noise_channel(epsilon):
    return lambda rho: w_noise_channel(rho=rho, alpha=(1 - epsilon))


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d)**2)


def _bool_from_probability(success_probability):
    if np.random.random() <= success_probability:
        return True
    else:
        return False


class ProbabilisticSource(SchedulingSource):
    def __init__(self, world, position, target_stations, trial_time, state_generation, check_success):
        if callable(trial_time):
            def time_distribution(source):
                times_tried = 1
                return trial_time(source), times_tried
        else:
            def time_distribution(source):
                times_tried = 1
                return trial_time, times_tried
        if callable(check_success):
            self.check_success = check_success
        else:
            self.check_success = lambda source: _bool_from_probability(success_probability=check_success)
        super(ProbabilisticSource, self).__init__(world=world, position=position,
                                                  target_stations=target_stations,
                                                  time_distribution=time_distribution,
                                                  state_generation=state_generation)

    def generate_pair(self, initial_state, initial_cost_add=None, initial_cost_max=None):
        if self.check_success(self):
            return super(ProbabilisticSource, self).generate_pair(initial_state=initial_state, initial_cost_add=initial_cost_add, initial_cost_max=initial_cost_max)
        else:
            pair = super(ProbabilisticSource, self).generate_pair(initial_state=initial_state, initial_cost_add=initial_cost_add, initial_cost_max=initial_cost_max)
            pair.destroy_and_track_resources()
            pair.qubits[0].destroy()
            pair.qubits[1].destroy()
            return None


class MultiMemoryProtocol(Protocol):
    def __init__(self, world, num_memories, mode="seq"):
        self.num_memories = num_memories
        self.mode = mode
        self.time_list = []
        self.fidelity_list = []
        self.correlations_z_list = []
        self.correlations_x_list = []
        self.resource_cost_max_list = []
        super(MultiMemoryProtocol, self).__init__(world=world)

    def setup(self):
        """Identifies the stations and sources in the world.

        Should be run after the relevant WorldObjects have been added
        to the world.

        Returns
        -------
        None

        """
        stations = self.world.world_objects["Station"]
        assert len(stations) == 3
        self.station_A, self.station_central, self.station_B = sorted(stations, key=lambda x: x.position)
        sources = self.world.world_objects["Source"]
        assert len(sources) == 2
        self.source_A = next(filter(lambda source: self.station_A in source.target_stations and self.station_central in source.target_stations, sources))
        self.source_B = next(filter(lambda source: self.station_central in source.target_stations and self.station_B in source.target_stations, sources))
        assert callable(getattr(self.source_A, "schedule_event", None))  # schedule_event is a required method for this protocol
        assert callable(getattr(self.source_B, "schedule_event", None))

    def _pair_is_between_stations(self, pair, station1, station2):
        return (pair.qubit1.station == station1 and pair.qubit2.station == station2) or (pair.qubit1.station == station2 and pair.qubit2.station == station1)

    def _get_left_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda x: self._pair_is_between_stations(x, self.station_A, self.station_central), pairs))

    def _get_right_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda x: self._pair_is_between_stations(x, self.station_central, self.station_B), pairs))

    def _get_long_range_pairs(self):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda x: self._pair_is_between_stations(x, self.station_A, self.station_B), pairs))

    def _left_pair_is_scheduled(self):
        try:
            next(filter(lambda event: (isinstance(event, SourceEvent)
                                       and (self.station_A in event.source.target_stations)
                                       and (self.station_central in event.source.target_stations)
                                       ),
                        self.world.event_queue.queue))
            return True
        except StopIteration:
            return False

    def _right_pair_is_scheduled(self):
        try:
            next(filter(lambda event: (isinstance(event, SourceEvent)
                                       and (self.station_central in event.source.target_stations)
                                       and (self.station_B in event.source.target_stations)
                                       ),
                        self.world.event_queue.queue))
            return True
        except StopIteration:
            return False

    def _eval_pair(self, long_range_pair):
        comm_distance = np.max([np.abs(self.station_central.position - self.station_A.position), np.abs(self.station_B.position - self.station_central.position)])
        comm_time = comm_distance / C

        pair_fidelity = np.real_if_close(np.dot(np.dot(mat.H(mat.phiplus), long_range_pair.state), mat.phiplus))[0, 0]
        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.fidelity_list += [pair_fidelity]

        z0z0 = mat.tensor(mat.z0, mat.z0)
        z1z1 = mat.tensor(mat.z1, mat.z1)
        correlations_z = np.real_if_close(np.dot(np.dot(mat.H(z0z0), long_range_pair.state), z0z0)[0, 0] + np.dot(np.dot(mat.H(z1z1), long_range_pair.state), z1z1))[0, 0]
        self.correlations_z_list += [correlations_z]

        x0x0 = mat.tensor(mat.x0, mat.x0)
        x1x1 = mat.tensor(mat.x1, mat.x1)
        correlations_x = np.real_if_close(np.dot(np.dot(mat.H(x0x0), long_range_pair.state), x0x0)[0, 0] + np.dot(np.dot(mat.H(x1x1), long_range_pair.state), x1x1))[0, 0]
        self.correlations_x_list += [correlations_x]

        self.resource_cost_max_list += [long_range_pair.resource_cost_max]
        return

    def check(self):
        if self.world.event_queue.queue:  # this protocol only acts when event_queue is empty
            return
        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        if self.mode == "sim":
            if num_left_pairs == 0:
                for _ in range(self.num_memories):
                    self.source_A.schedule_event()
            if num_right_pairs == 0:
                for _ in range(self.num_memories):
                    self.source_B.schedule_event()
        elif self.mode == "seq":
            if num_left_pairs == 0:
                for _ in range(self.num_memories):
                    self.source_A.schedule_event()
            if num_left_pairs != 0 and num_right_pairs == 0:
                for _ in range(self.num_memories):
                    self.source_B.schedule_event()
        if num_left_pairs != 0 and num_right_pairs != 0:
            num_swappings = min(num_left_pairs, num_right_pairs)
            # discard leftover pairs
            for pair in left_pairs[num_swappings:] + right_pairs[num_swappings:]:
                pair.destroy_and_track_resources()
                pair.qubits[0].destroy()
                pair.qubits[1].destroy()
            for left_pair, right_pair in zip(left_pairs[:num_swappings], right_pairs[:num_swappings]):
                ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pair, right_pair])
                self.world.event_queue.add_event(ent_swap_event)
        long_range_pairs = self._get_long_range_pairs()
        if long_range_pairs:
            for long_range_pair in long_range_pairs:
                self._eval_pair(long_range_pair)
                # cleanup
                long_range_pair.qubits[0].destroy()
                long_range_pair.qubits[1].destroy()
                long_range_pair.destroy()
            self.check()


def run(length, max_iter, params, num_memories, mode="seq"):
    cutoff_time = None  # cutoff currently not supported by protocol
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

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def trial_time(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        return T_P + 2 * comm_distance / C

    def check_success(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        return _bool_from_probability(eta_effective)

    def state_generation(source):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # dephasing that has accrued while other qubit was travelling
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
    station_B = Station(world, position=length, memory_noise=None,
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    station_central = Station(world, id=2, position=length / 2,
                              memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP),
                              memory_cutoff_time=cutoff_time,
                              BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                              )
    source_A = ProbabilisticSource(world, position=length / 2, target_stations=[station_A, station_central], trial_time=trial_time, state_generation=state_generation, check_success=check_success)
    source_B = ProbabilisticSource(world, position=length / 2, target_stations=[station_central, station_B], trial_time=trial_time, state_generation=state_generation, check_success=check_success)
    protocol = MultiMemoryProtocol(world, num_memories=num_memories)
    protocol.setup()

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    p = run(length=22000, max_iter=1000, params={}, num_memories=100, mode="seq")
    import matplotlib.pyplot as plt
    plt.scatter(p.time_list, p.resource_cost_max_list)
    plt.show()
