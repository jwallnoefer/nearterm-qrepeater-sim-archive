from events import EventQueue


class World(object):
    """A collection of WorldObjects with an EventQueue.

    The World can be understood as a central object describing an experimental
    setup. It keeps track of all WorldObjects, which contain all the information
    a protocol can access to make decisions.

    Attributes
    ----------
    event_queue : EventQueue
        A schedule of events so they can be resolved in order.
    world_objects : dict of str: WorldObject
        keys: str of WorldObject subclass names (obtained via x.__class__.__name__)
        values: list of WorldObjects

    """

    def __init__(self):
        self.event_queue = EventQueue()
        self.world_objects = {}  # should contain everything about the state world that the protocol needs to know

    def __contains__(self, world_object):
        return world_object in self.world_objects[world_object.type]

    def print_status(self):
        """Print scheduled events and objects in a somewhat formatted way.

        Returns
        -------
        None

        """
        print("Event queue:")
        for event in self.event_queue.queue:
            print(event)
        print("%================%")
        print("Objects")
        for k, v in self.world_objects.items():
            print("------")
            print(k + ":")
            for obj in v:
                print(obj)

    def register_world_object(self, world_object):
        """Add a WorldObject to this world.

        Parameters
        ----------
        world_object : WorldObject
            WorldObject that should be added and tracked by `self.world_objects`

        Returns
        -------
        None

        """
        object_type = world_object.type
        if object_type not in self.world_objects:
            self.world_objects[object_type] = []
        self.world_objects[object_type] += [world_object]

    def deregister_world_object(self, world_object):
        """Remove a WorldObject from this World.

        Parameters
        ----------
        world_object : WorldObject
            The WorldObject that is removed from this World.

        Returns
        -------
        None

        """
        object_type = world_object.type
        type_list = self.world_objects[object_type]
        type_list.remove(world_object)
