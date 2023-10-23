"""A simplified scenario with only very basic parameters."""
import os, sys;
from abc import abstractmethod

sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from functools import lru_cache
from requsim.world import World
from requsim.quantum_objects import Station, SchedulingSource
from requsim.events import SourceEvent, EntanglementSwappingEvent
from requsim.noise import NoiseModel, NoiseChannel
from requsim.tools.protocol import Protocol
from consts import SPEED_OF_LIGHT_IN_OPTICAL_FIBER as C
from consts import ATTENUATION_LENGTH_IN_OPTICAL_FIBER as L_ATT
from collections import defaultdict
from warnings import warn
from requsim.libs.aux_functions import apply_single_qubit_map, distance
from requsim.tools.noise_channels import y_noise_channel, z_noise_channel, w_noise_channel
import requsim.libs.matrix as mat


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


@lru_cache(maxsize=int(5e4))
def is_event_swapping_pairs(event, pair1, pair2):
    return isinstance(event, EntanglementSwappingEvent) and (pair1 in event.pairs) and (pair2 in event.pairs)


@lru_cache(maxsize=int(5e4))
def is_sourceevent_between_stations(event, station1, station2):
    return isinstance(event, SourceEvent) and (station1 in event.source.target_stations) and (station2 in event.source.target_stations)


class BaseManylinkProtocol(Protocol):
    def __init__(self, world, stations, sources, num_memories=1, communication_speed=C):
        self.stations = stations
        self.sources = sources
        self.num_memories = num_memories
        self.communication_speed = communication_speed
        self.time_list = []
        self.state_list = []
        self.scheduled_swappings = defaultdict(lambda: [])
        super(BaseManylinkProtocol, self).__init__(world=world)

    def setup(self):
        # Station ordering left to right
        num_stations = len(self.stations)
        num_sources = len(self.sources)
        assert num_sources == num_stations - 1
        for source in self.sources:
            assert callable(getattr(source, "schedule_event", None))  # schedule_event is a required method for this protocol
        self.link_stations = [[self.stations[i], self.stations[i + 1]] for i in range(num_sources)]
        self.host_station_by_source = {}
        for i, source in enumerate(self.sources):
            self.host_station_by_source[source] = self.stations[2 * (i // 2) + 1]
        self.sources_by_station = defaultdict(lambda: [])
        for source in self.sources:
            self.sources_by_station[source.target_stations[0]] += [source]
            self.sources_by_station[source.target_stations[1]] += [source]

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list})

    def _get_pairs_between_stations(self, station1, station2):
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        return list(filter(lambda pair: pair.is_between_stations(station1, station2), pairs))

    def _get_pairs_scheduled(self, station1, station2):
        return list(filter(lambda event: is_sourceevent_between_stations(event, station1, station2),
                    self.world.event_queue.queue))

    def _eval_pair(self, long_range_pair):
        comm_distance = np.max([distance(self.stations[0], self.stations[-2]),
                                distance(self.stations[-1], self.stations[1])])
        # comm_distance is simple upper limit for swapping communication
        comm_time = comm_distance / self.communication_speed

        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_pair.state]
        return

    def pairs_at_station(self, station):
        station_index = self.stations.index(station)
        pairs_left = []
        pairs_right = []
        for qubit in station.qubits:
            pair = qubit.higher_order_object
            qubit_list = list(pair.qubits)
            qubit_list.remove(qubit)
            qubit_neighbor = qubit_list[0]
            if self.stations.index(qubit_neighbor._info["station"]) < station_index:
                pairs_left += [pair]
            else:
                pairs_right += [pair]
        return (pairs_left, pairs_right)

    def memory_check(self, station):
        station_index = self.stations.index(station)
        free_memories_left = self.num_memories
        free_memories_right = self.num_memories
        pairs_left, pairs_right = self.pairs_at_station(station)
        free_memories_left -= len(pairs_left)
        free_memories_right -= len(pairs_right)
        free_memories_left -= len(self._get_pairs_scheduled(self.stations[station_index - 1], station))
        free_memories_right -= len(self._get_pairs_scheduled(station, self.stations[station_index + 1]))
        return (free_memories_left, free_memories_right)

    def _check_station_overflow(self, station):
        left_pairs, right_pairs = self.pairs_at_station(station)
        has_overflowed = False
        if len(left_pairs) > self.num_memories:
            last_pair = left_pairs[-1]
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy()
            has_overflowed = True
        if len(right_pairs) > self.num_memories:
            last_pair = right_pairs[-1]
            last_pair.qubits[0].destroy()
            last_pair.qubits[1].destroy()
            last_pair.destroy()
            has_overflowed = True
        return has_overflowed

    def _check_new_source_events(self, station):
        sources_to_check = self.sources_by_station[station]
        for source in sources_to_check:
            host_station = self.host_station_by_source[source]
            free_left, free_right = self.memory_check(host_station)
            for _ in range(free_left):
                self.sources_by_station[host_station][0].schedule_event()
            for _ in range(free_right):
                self.sources_by_station[host_station][1].schedule_event()

    def _check_swapping(self, station):
        left_pairs, right_pairs = self.pairs_at_station(station)
        num_swappings = min(len(left_pairs), len(right_pairs))
        if num_swappings:
            # get rid of events that are no longer scheduled
            self.scheduled_swappings[station] = [event for event in self.scheduled_swappings[station] if
                                                 event in self.world.event_queue.queue]
        for left_pair, right_pair in zip(left_pairs[:num_swappings], right_pairs[:num_swappings]):
            # assert that we do not schedule the same swapping more than once
            try:
                next(filter(lambda event: is_event_swapping_pairs(event, left_pair, right_pair),
                            self.scheduled_swappings[station]))
                is_already_scheduled = True
            except StopIteration:
                is_already_scheduled = False
            if not is_already_scheduled:
                ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time,
                                                           pairs=[left_pair, right_pair],
                                                           station=station
                                                           )
                self.scheduled_swappings[station] += [ent_swap_event]
                self.world.event_queue.add_event(ent_swap_event)

    def _check_long_distance_pair(self):
        # Evaluate long range pairs
        long_range_pairs = self._get_pairs_between_stations(self.stations[0], self.stations[-1])
        if long_range_pairs:
            for long_range_pair in long_range_pairs:
                self._eval_pair(long_range_pair)
                # cleanup
                long_range_pair.qubits[0].destroy()
                long_range_pair.qubits[1].destroy()
                long_range_pair.destroy()
            # self.check()  # was useful at some point for other scenarios

    @abstractmethod
    def check(self, message=None):
        raise NotImplementedError


class DefaultManylinkProtocol(BaseManylinkProtocol):
    def check(self, message=None):
        if message is None:
            for station in self.stations:
                self._check_station_overflow(station)
            for station in self.stations:
                self._check_new_source_events(station)
            for station in self.stations:
                self._check_swapping(station)
            self._check_long_distance_pair()
        elif message["event_type"] == "SourceEvent" and message["resolve_successful"] is True:
            output_pair = message["output_pair"]
            stations = [output_pair.qubit1._info["station"], output_pair.qubit2._info["station"]]
            for station in stations:
                has_overflowed = self._check_station_overflow(station)
                if has_overflowed:
                    self._check_new_source_events(station)
                self._check_swapping(station)
        elif message["event_type"] == "SourceEvent" and message["resolve_successful"] is False:
            warn("A SourceEvent has resolved unsuccessfully. This should never happen.")
        elif message["event_type"] == "DiscardQubitEvent" and message["resolve_successful"] is True:
            discarded_qubit = message["qubit"]
            self._check_new_source_events(discarded_qubit._info["station"])
        elif message["event_type"] == "DiscardQubitEvent" and message["resolve_successful"] is False:
            pass
        elif message["event_type"] == "EntanglementSwappingEvent" and message["resolve_successful"] is True:
            self._check_new_source_events(message["swapping_station"])
            output_pair = message["output_pair"]
            for station in [output_pair.qubit1._info["station"], output_pair.qubit2._info["station"]]:
                self._check_swapping(station)
            self._check_long_distance_pair()
        elif message["event_type"] == "EntanglementSwappingEvent" and message["resolve_successful"] is False:
            # warn("An EntanglementSwappingEvent has resolved unsuccessfully. Trying to recover.")
            # for station in self.stations:
            #     self._check_swapping(station)
            pass
        else:
            warn(f"Unrecognized message type encountered: {message}")

class CustomManylinkProtocol(BaseManylinkProtocol):
    def __init__(self, world, stations, sources, num_memories=1, communication_speed=C):
        self.step = 0
        self.counter = 0
        super(CustomManylinkProtocol, self).__init__(world=world, stations=stations, sources=sources, num_memories=num_memories, communication_speed=communication_speed)


    def check(self, message=None):
        if len(self.world.event_queue.queue) != 0:
            return
        if self.step == 0:
            for source in self.sources:
                source.schedule_event()
            self.step = 1
        elif self.step == 1:
            pairs = self.world.world_objects["Pair"]
            if len(pairs) == 1:
                self._check_long_distance_pair()
                self.step = 0
                self.counter = 0
                self.check()
            else:
                self._check_swapping(self.stations[1 + self.counter])
                self.counter += 1


def run(length, max_iter, params, num_links, protocol_class):
    assert num_links % 2 == 0
    allowed_params = ["F_INIT", "T_DP"]
    P_LINK = 1
    P_D = 0
    T_P = 0
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")
    # unpack the parameters
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(e.__traceback__)

    def time_distribution(source):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position),
                                np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        eta_effective = 1 - (1 - eta) * (1 - P_D) ** 2
        trial_time = T_P + comm_time  # I don't think that paper uses latency time and loading time?
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time

    @lru_cache()
    def state_generation(source):
        state = F_INIT * (mat.phiplus @ mat.H(mat.phiplus)) + \
                (1 - F_INIT) / 3 * (mat.psiplus @ mat.H(mat.psiplus) +
                                    mat.phiminus @ mat.H(mat.phiminus) +
                                    mat.psiminus @ mat.H(mat.psiminus)
                                    )
        comm_distance = np.max([distance(source, source.target_stations[0]), distance(source.target_stations[1], source)])
        trial_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:  # dephasing that has accrued while other qubit was travelling
                storage_time = trial_time - distance(source, station) / C  # qubit is in storage for varying amounts of time
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
        return state

    station_positions = [x * length / num_links for x in range(num_links + 1)]

    world = World()
    station_A = Station(world, position=station_positions[0], memory_noise=None)
    other_stations = [Station(world, position=pos,
                              memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP)
                              )
                      for pos in station_positions[1:-1]
                      ]
    station_B = Station(world, position=station_positions[-1], memory_noise=None)
    stations = [station_A] + other_stations + [station_B]
    source_positions = [station_positions[2 * (i // 2) + 1] for i in range(num_links)]
    sources = []
    for i, source_position in enumerate(source_positions):
        sources += [SchedulingSource(world, position=source_position,
                                     target_stations=(stations[i], stations[i + 1]),
                                     time_distribution=time_distribution,
                                     state_generation=state_generation)
                    ]
    protocol = protocol_class(world, stations, sources, num_memories=1, communication_speed=C)
    protocol.setup()

    # from code import interact
    # interact(local=locals())
    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(current_message)
        try:
            current_message = world.event_queue.resolve_next_event()
        except IndexError:
            world.print_status()
            from code import interact
            interact(local=locals())

    return protocol


if __name__ == "__main__":
    from time import time
    max_iter = 10
    # x = np.linspace(0, 1024, num=64 + 1, dtype=int)[1:]
    x = np.linspace(0, 1024, num=8 + 1, dtype=int)[1:]
    y1 = []
    y2 = []
    for num_links in x:
        start_time = time()
        res = run(length=22000, max_iter=max_iter,
                  params={"T_DP": 25, "F_INIT": 0.95},
                  num_links=num_links, protocol_class=DefaultManylinkProtocol)
        time_interval = (time() - start_time) / max_iter
        y1 += [time_interval]
        print(num_links, f"{time_interval:.2f}")
    for num_links in x:
        start_time = time()
        res = run(length=22000, max_iter=max_iter,
                  params={"T_DP": 25, "F_INIT": 0.95},
                  num_links=num_links, protocol_class=CustomManylinkProtocol)
        time_interval = (time() - start_time) / max_iter
        y2 += [time_interval]
        print(num_links, f"{time_interval:.2f}")
    import matplotlib.pyplot as plt
    plt.plot(x, y1, label="default")
    plt.plot(x, y2, label="custom")
    plt.xlabel("num_links")
    plt.ylabel("time [s]")
    # plt.yscale("log")
    plt.grid()
    plt.show()
