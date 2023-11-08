
class SingletonByNode:
    _instances = {}  # Store instances based on node name

    @classmethod
    def instance(cls, node):
        node_name = node  # Replace with your own logic to get unique node name
        if node_name not in cls._instances:
            cls._instances[node_name] = cls(node)
        return cls._instances[node_name]