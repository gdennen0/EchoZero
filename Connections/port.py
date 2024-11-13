from connection import Connection
from message import Log


class Port:
    def __init__(self, type, name="default"):
        self.type = type
        self.name = name
        self.connections =  []

    def connect(self, port):
        connection = Connection(self, port)
        self.connections.append(connection)
        Log.info(f"'{self.name}' Port connected to '{port.name}' Port")

    def disconnect(self, connection):
        self.connections.remove(connection) 
        Log.info(f"'{self.name}' Port disconnected from '{connection.output_port.name}' Port")

    def list_connections(self):
        for connection in self.connections:
            Log.info(f"'{self.name}' Port connected to '{connection.output_port.name}' Port")


