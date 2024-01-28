import pickle
import socket
import struct
import threading
import traceback

from magicroto.utils.logger import logger


class ServerClientBase:

    def __init__(self):
        # Stores client threads and connections
        self.client_threads = {}
        self.client_connections = {}

    def close_connection(self, addr):
        return NotImplementedError

    def close_server(self):
        return NotImplementedError

    @classmethod
    def encode_data(cls, data):
        serialized_data = pickle.dumps(data)
        payload_size = struct.pack('!I', len(serialized_data))
        return b'data::' + payload_size + serialized_data

    @classmethod
    def decode_data(cls, data_buffer):
        payload_data = b""
        payload_size = None

        while True:
            # Check for the header and split the buffer accordingly
            if data_buffer and b'::' in data_buffer and payload_size is None:
                header, data_buffer = data_buffer.split(b'::', 1)

                if header == b'data':
                    # Extract payload size and update the buffer
                    payload_size = struct.unpack('!I', data_buffer[:4])[0]
                    data_buffer = data_buffer[4:]

            # Accumulate the payload data
            if payload_size is not None:
                payload_data += data_buffer
                data_buffer = b""

                # Check if the complete payload is received
                if len(payload_data) >= payload_size:
                    actual_data = payload_data[:payload_size]

                    # Deserialize the actual data
                    try:
                        decoded_data = pickle.loads(actual_data)
                        return decoded_data
                    except Exception as e:
                        # Handle deserialization error
                        raise Exception(f"An error occurred while decoding data: {e}")

            # If no complete data is received yet, break the loop
            if not data_buffer:
                break

        # Return None or raise an error if the complete data is not received
        return None

    def receive_data(self, conn, addr, data_handler=None, return_response_data=False):
        logger.debug(f"Inside receive_data for {addr}.")

        data_buffer = b""
        payload_data = b""
        payload_size = None
        handler = data_handler

        while True:
            try:
                packet = conn.recv(4096)
                if not packet:
                    break
                data_buffer += packet

                while b'::' in data_buffer and payload_size is None:
                    header, data_buffer = data_buffer.split(b'::', 1)
                    if header == b'command':
                        if data_buffer == b'quit':
                            conn.sendall(b'ack')
                            self.close_server()
                            return

                    elif header == b'data':
                        payload_size = struct.unpack('!I', data_buffer[:4])[0]
                        data_buffer = data_buffer[4:]

                if payload_size is not None:
                    payload_data += data_buffer
                    data_buffer = b""

                    if len(payload_data) >= payload_size:
                        actual_data = payload_data[:payload_size]
                        decoded_data = None
                        # First try block to check data loading
                        try:
                            decoded_data = pickle.loads(actual_data)
                        except Exception as e:
                            logger.error(f"An error occurred while decoding data: {e}")

                        # Second try block to check data_handler
                        if decoded_data is not None:
                            try:
                                response_data = None
                                if handler:
                                    response_data = handler(decoded_data)

                                # Sending an acknowledgment back to the client
                                if response_data is not None and return_response_data:
                                    encoded_response = SocketServer.encode_data(response_data)
                                    logger.debug(f'Sending encoded_response type:{type(response_data)}')
                                    conn.sendall(encoded_response)
                                else:
                                    conn.sendall(b'ack')
                            except Exception as e:
                                tb = traceback.format_exc()
                                logger.error(f"An error occurred in data_handler: {e}\n{tb}")

                        data_buffer = payload_data[payload_size:]
                        payload_data = b""
                        payload_size = None

            except ConnectionResetError:
                logger.error("Connection was closed by the client.")
                self.close_connection(addr)
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                break

        # Clean up and close the connection


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        port = kwargs.get('port', None)
        if port is not None:
            if port not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[port] = instance
            return cls._instances[port]
        else:
            return super().__call__(*args, **kwargs)


class SocketServer(ServerClientBase, metaclass=SingletonMeta):
    @classmethod
    def get_instances(cls):
        """
        Returns a copy of the dictionary containing all initialized instances
        of the SocketServer class.
        """
        return cls._instances.copy()

    def __init__(self, port=None, data_handler=None, host="localhost"):
        super().__init__()
        self.data_handler = data_handler
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if port:
            self.port = port
        else:
            self.port = self.find_available_port()

        self.server_socket.bind((host, self.port))
        self.server_socket.listen(1)
        handler_logger = f', DataHandler:{data_handler.__name__}' if data_handler else ''
        logger.info(f"Server created on port {self.port}, Host:{host}{handler_logger}")

    @staticmethod
    def find_available_port():
        temp_sock = socket.socket()
        temp_sock.bind(('', 0))
        port = temp_sock.getsockname()[1]
        temp_sock.close()
        return port

    def start_accepting_clients(self, data_handler=None, return_response_data=False):
        logger.info(f"Server starting on port {self.port}")
        print(f"Server starting on port {self.port}")
        data_handler = data_handler or self.data_handler

        thread = threading.Thread(target=self.accept_client, args=(data_handler, return_response_data,))
        thread.start()
        logger.debug(f"Starting to accept clients. on {thread}")

    def accept_client(self, data_handler=None, return_response_data=False):
        logger.debug("Inside accept_client.")
        while True:
            conn, addr = self.server_socket.accept()
            logger.debug(f"Accepted connection from {addr}.")

            # Store the client connection
            self.client_connections[addr] = conn

            # Start a new thread for each client
            thread = threading.Thread(target=self.receive_data, args=(conn, addr, data_handler, return_response_data,))
            thread.start()

            # Store the thread
            self.client_threads[addr] = thread

    def send_to_all_clients(self, data):
        """
        Sends data to all connected clients.

        @param data: The data to be sent to the clients.
        """
        encoded_data = self.encode_data(data)
        for addr, conn in self.client_connections.items():
            try:
                conn.sendall(encoded_data)
                logger.info(f"Data sent to client at {addr}")
            except Exception as e:
                logger.error(f"Failed to send data to client at {addr}: {e}")

    def close_connection(self, addr):
        logger.debug(f"Closing connection with {addr}")
        if addr in self.client_connections:
            self.client_connections[addr].close()
            del self.client_connections[addr]
            del self.client_threads[addr]
            logger.info(f"Server is closed {addr}")

    def close_server(self):
        """
        Closes all client connections and then shuts down the server.
        """
        # Close all client connections
        for addr, conn in self.client_connections.items():
            try:
                conn.close()
                logger.info(f"Closed connection with client at {addr}")
            except Exception as e:
                logger.error(f"Error closing connection with client at {addr}: {e}")

        # Clear the connections and threads dictionaries
        self.client_connections.clear()
        self.client_threads.clear()

        # Shut down and close the server socket if it's open
        if self.server_socket.fileno() != -1:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except Exception as e:
                logger.error(f"Error shutting down server socket: {e}")

            try:
                self.server_socket.close()
                logger.info("Server socket closed successfully.")
            except Exception as e:
                logger.error(f"Error closing server socket: {e}")

# if __name__ == '__main__':
#     # Initialize SocketServer
#     server = SocketServer(port=5050, data_handler=print_received_data)
#
#     # Accept a client connection
#     server.start_accepting_clients()
