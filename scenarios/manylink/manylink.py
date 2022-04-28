import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from functools import lru_cache
from events import SourceEvent, EntanglementSwappingEvent
from protocol import MessageReadingProtocol
from consts import SPEED_OF_LIGHT_IN_OPTICAL_FIBER as C
from libs.aux_functions import apply_single_qubit_map, w_noise_channel, distance


@lru_cache(maxsize=int(1e6))
def is_event_swapping_pairs(event, pair1, pair2):
    return isinstance(event, EntanglementSwappingEvent) and (pair1 in event.pairs) and (pair2 in event.pairs)


@lru_cache(maxsize=int(1e6))
def is_sourceevent_between_stations(event, station1, station2):
    return isinstance(event, SourceEvent) and (station1 in event.source.target_stations) and (station2 in event.source.target_stations)


class ManylinkProtocol(MessageReadingProtocol):
    def __init__(self, world, stations, sources, num_memories=1, communication_speed=C):
        self.stations = stations
        self.sources = sources
        self.num_memories = num_memories
        self.communication_speed = communication_speed
        self.time_list = []
        self.state_list = []
        self.resource_cost_max_list = []
        self.resource_cost_add_list = []
        super(ManylinkProtocol, self).__init__(world=world)

    def setup(self):
        # Station ordering left to right
        num_stations = len(self.stations)
        num_sources = len(self.sources)
        assert num_sources == num_stations - 1
        for source in self.sources:
            assert callable(getattr(source, "schedule_event", None))  # schedule_event is a required method for this protocol
        self.link_stations = [[self.stations[i], self.stations[i + 1]] for i in range(num_sources)]

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list,
                             "resource_cost_max": self.resource_cost_max_list,
                             "resource_cost_add": self.resource_cost_add_list})

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
        comm_distance = np.max([distance(self.sat_central, self.station_ground_left), distance(self.sat_central, self.station_ground_right)])
        comm_time = comm_distance / self.communication_speed

        self.time_list += [self.world.event_queue.current_time + comm_time]
        self.state_list += [long_range_pair.state]
        self.resource_cost_max_list += [long_range_pair.resource_cost_max]
        self.resource_cost_add_list += [long_range_pair.resource_cost_add]
        return
