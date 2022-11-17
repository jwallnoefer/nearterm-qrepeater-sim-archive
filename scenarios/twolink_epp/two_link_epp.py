import os, sys; sys.path.insert(0, os.path.abspath("."))
from requsim.quantum_objects import SchedulingSource, Station
from requsim.tools.protocol import TwoLinkProtocol
from requsim.world import World
from requsim.events import EntanglementSwappingEvent, EntanglementPurificationEvent
import requsim.libs.matrix as mat
import numpy as np
from requsim.libs.aux_functions import apply_single_qubit_map, distance
from requsim.tools.noise_channels import y_noise_channel, z_noise_channel, w_noise_channel
from warnings import warn
from collections import defaultdict
from requsim.noise import NoiseModel, NoiseChannel
from consts import SPEED_OF_LIGHT_IN_OPTICAL_FIBER as C
from consts import ATTENUATION_LENGTH_IN_OPTICAL_FIBER as L_ATT

# NOTE: this scenario uses the newer requsim package instead of the outdated
#       code in this repo, because we need one particular functionality here
#       requsim is available as a python package on PyPI
#       Source Code: https://github.com/jwallnoefer/requsim


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return NoiseChannel(n_qubits=1, channel_function=dephasing_noise_channel)


def construct_y_noise_channel(epsilon):
    return lambda rho: y_noise_channel(rho=rho, epsilon=epsilon)


def construct_w_noise_channel(epsilon):
    return lambda rho: w_noise_channel(rho=rho, alpha=(1 - epsilon))


def alpha_of_eta(eta, p_d):
    return eta * (1 - p_d) / (1 - (1 - eta) * (1 - p_d)**2)


class TwoLinkOneStepEPP(TwoLinkProtocol):
    def __init__(self, world, num_memories=2, epp_steps=1, communication_speed=C, measure_asap=True):
        self.epp_tracking = defaultdict(lambda: 0)
        self.num_memories = num_memories
        self.epp_steps = epp_steps
        self.measure_asap = measure_asap
        super(TwoLinkOneStepEPP, self).__init__(world, communication_speed=communication_speed)

    def _left_epp_is_scheduled(self):
        try:
            next(filter(lambda event: (isinstance(event, EntanglementPurificationEvent)
                                       and (self.station_A in [qubit._info["station"] for qubit in event.pairs[0].qubits])
                                       and (self.station_central in [qubit._info["station"] for qubit in event.pairs[0].qubits])
                                       ),
                        self.world.event_queue.queue))
            return True
        except StopIteration:
            return False

    def _right_epp_is_scheduled(self):
        try:
            next(filter(lambda event: (isinstance(event, EntanglementPurificationEvent)
                                       and (self.station_central in [qubit._info["station"] for qubit in event.pairs[0].qubits])
                                       and (self.station_B in [qubit._info["station"] for qubit in event.pairs[0].qubits])
                                       ),
                        self.world.event_queue.queue))
            return True
        except StopIteration:
            return False

    def _remove_endstation_noise(self, pair):
        for qubit in pair.qubits:
            if qubit in self.station_A.qubits and self.station_A.memory_noise is not None:
                qubit.remove_time_dependent_noise(self.station_A.memory_noise)
            elif qubit in self.station_B.qubits and self.station_B.memory_noise is not None:
                qubit.remove_time_dependent_noise(self.station_B.memory_noise)

    def check(self, message=None):
        """Checks world state and schedules new events.

        Returns
        -------
        None

        """
        try:
            all_pairs = self.world.world_objects["Pair"]
        except KeyError:
            all_pairs = []

        if message is not None and message["event_type"] == "EntanglementPurificationEvent":
            if message["is_successful"]:
                output_pair = message["output_pair"]
                self.epp_tracking[output_pair] += 1
                if self.epp_tracking[output_pair] == self.epp_steps and self.measure_asap:
                    self._remove_endstation_noise(output_pair)

        for pair in list(self.epp_tracking):
            if pair not in all_pairs:
                self.epp_tracking.pop(pair)

        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        num_left_pairs_scheduled = len(self._left_pairs_scheduled())
        num_right_pairs_scheduled = len(self._right_pairs_scheduled())
        left_used = num_left_pairs + num_left_pairs_scheduled
        right_used = num_right_pairs + num_right_pairs_scheduled
        # schedule pair creation if a memory is not busy
        if left_used < self.num_memories:
            for _ in range(self.num_memories - left_used):
                self.source_A.schedule_event()
        if right_used < self.num_memories:
            for _ in range(self.num_memories - right_used):
                self.source_B.schedule_event()

        # check for epp
        if num_left_pairs >= 2 and not self._left_epp_is_scheduled():
            # group pairs by epp step
            staged_pairs = defaultdict(list)
            for pair in left_pairs:
                if not pair.is_blocked:
                    steps = self.epp_tracking[pair]
                    staged_pairs[steps] += [pair]
            # find purifiable pairs at same recurrence level
            for steps, pairs in staged_pairs.items():
                if len(pairs) >= 2 and steps < self.epp_steps:
                    communcation_time = distance(self.station_A, self.station_central) / self.communication_speed
                    epp_event = EntanglementPurificationEvent(time=self.world.event_queue.current_time,
                                                              pairs=pairs[0:2],
                                                              protocol="dejmps",
                                                              communication_time=communcation_time)
                    self.world.event_queue.add_event(epp_event)

        if num_right_pairs >= 2 and not self._right_epp_is_scheduled():
            # group pairs by epp step
            staged_pairs = defaultdict(list)
            for pair in right_pairs:
                if not pair.is_blocked:
                    steps = self.epp_tracking[pair]
                    staged_pairs[steps] += [pair]
            # find purifiable pairs at same recurrence level
            for steps, pairs in staged_pairs.items():
                if len(pairs) >= 2 and steps < self.epp_steps:
                    communcation_time = distance(self.station_central, self.station_B) / self.communication_speed
                    epp_event = EntanglementPurificationEvent(time=self.world.event_queue.current_time,
                                                              pairs=pairs[0:2],
                                                              protocol="dejmps",
                                                              communication_time=communcation_time)
                    self.world.event_queue.add_event(epp_event)

        # check for swapping
        if num_left_pairs >= 1 and num_right_pairs >= 1:
            left_swap_ready = list(filter(lambda x: self.epp_tracking[x] == self.epp_steps and not x.is_blocked, left_pairs))
            right_swap_ready = list(filter(lambda x: self.epp_tracking[x] == self.epp_steps and not x.is_blocked, right_pairs))
            if left_swap_ready and right_swap_ready:
                left_pair = left_swap_ready[0]
                right_pair = right_swap_ready[0]
                # assert that we do not schedule the same swapping more than once
                try:
                    next(filter(lambda event: (isinstance(event, EntanglementSwappingEvent)
                                               and (left_pair in event.pairs)
                                               and (right_pair in event.pairs)
                                               ),
                                self.world.event_queue.queue))
                    is_already_scheduled = True
                except StopIteration:
                    is_already_scheduled = False
                if not is_already_scheduled:
                    ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time,
                                                               pairs=[left_pair, right_pair],
                                                               station=self.station_central)
                    self.world.event_queue.add_event(ent_swap_event)

        long_range_pairs = self._get_long_range_pairs()
        for long_range_pair in long_range_pairs:
            self._eval_pair(long_range_pair)
            # cleanup
            long_range_pair.qubits[0].destroy()
            long_range_pair.qubits[1].destroy()
            long_range_pair.destroy()

        if not self.world.event_queue.queue:
            warn("Protocol may be stuck in a state without events.")


def run(length, max_iter, params, cutoff_time=None, num_memories=2, epp_steps=1, measure_asap=True):
    allowed_params = ["P_LINK", "T_P", "E_MA", "P_D", "LAMBDA_BSM", "F_INIT", "T_DP"]
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")
    # unpack the parameters
    P_LINK = params.get("P_LINK", 1.0)
    T_P = params.get("T_P", 0)  # preparation time
    E_MA = params.get("E_MA", 0)  # misalignment error
    P_D = params.get("P_D", 0)  # dark count probability
    LAMBDA_BSM = params.get("LAMBDA_BSM", 1)  # Bell state measurement ideality parameter
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(e.__traceback__)

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def time_distribution(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time#, random_num

    def state_generation(source):
        state = F_INIT * (mat.phiplus @ mat.H(mat.phiplus)) + \
                (1 - F_INIT) / 3 * (mat.psiplus @ mat.H(mat.psiplus) +
                                    mat.phiminus @ mat.H(mat.phiminus) +
                                    mat.psiminus @ mat.H(mat.psiminus)
                                    )
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        trial_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # dephasing that has accrued while other qubit was travelling
                storage_time = trial_time - distance(source, station) / C  # qubit is in storage for varying amounts of time
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
            if station.dark_count_probability is not None:  # dark counts are handled here because the information about eta is needed for that
                eta = P_LINK * np.exp(-comm_distance / L_ATT)
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta=eta, p_d=station.dark_count_probability))
        return state

    misalignment_noise = NoiseChannel(n_qubits=1, channel_function=construct_y_noise_channel(epsilon=E_MA))

    world = World()
    if epp_steps == 0 and measure_asap is True:
        def end_station_noise():
            return None
    else:
        def end_station_noise():
            return construct_dephasing_noise_channel(dephasing_time=T_DP)
    station_A = Station(world, position=0, memory_noise=end_station_noise(),
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    station_central = Station(world, position=length / 2,
                              memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP),
                              memory_cutoff_time=cutoff_time,
                              BSM_noise_model=NoiseModel(channel_before=NoiseChannel(n_qubits=4, channel_function=imperfect_bsm_err_func))
                              )
    station_B = Station(world, position=length, memory_noise=end_station_noise(),
                        creation_noise_channel=misalignment_noise,
                        dark_count_probability=P_D
                        )
    source_A = SchedulingSource(world, position=length / 2, target_stations=[station_A, station_central], time_distribution=time_distribution, state_generation=state_generation)
    source_B = SchedulingSource(world, position=length / 2, target_stations=[station_central, station_B], time_distribution=time_distribution, state_generation=state_generation)
    protocol = TwoLinkOneStepEPP(world, num_memories=num_memories, epp_steps=epp_steps)
    protocol.setup()

    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(current_message)
        current_message = world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    from time import time
    start_time = time()
    max_iter = 2500
    # np.random.seed(14725234)
    res = run(length=1e3, max_iter=max_iter, params={"P_LINK": 0.5, "T_DP": 1.0, "F_INIT": 0.925, "P_D": 1e-6}, cutoff_time=400e-3, num_memories=2, epp_steps=1)
    print(res.data)
    res.world.print_status()
    res.world.event_queue.print_stats()
    print(f"{max_iter} datapoints obtained in {time()-start_time:.2f} seconds")
