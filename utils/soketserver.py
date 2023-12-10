import pickle
import socket
import struct
import threading
import traceback

from magicroto.utils.logger import logger


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


class SocketServer(metaclass=SingletonMeta):
    @classmethod
    def get_instances(cls):
        """
        Returns a copy of the dictionary containing all initialized instances
        of the SocketServer class.
        """
        return cls._instances.copy()

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

    def __init__(self, port=None, data_handler=None, host="localhost"):

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
        thread = threading.Thread(target=self.accept_client, args=(data_handler, return_response_data,))
        thread.start()
        logger.debug(f"Starting to accept clients. on {thread}")

    def accept_client(self, data_handler=None, return_response_data=False):
        logger.debug("Inside accept_client.")
        self.conn, self.addr = self.server_socket.accept()
        logger.debug(f"Accepted connection from {self.addr}.")
        thread = threading.Thread(target=self.receive_data, args=(data_handler, return_response_data,))
        thread.start()

    def receive_data(self, data_handler=None, return_response_data=False):
        logger.debug("Inside receive_data.")

        data_buffer = b""
        payload_data = b""
        payload_size = None
        handler = data_handler if data_handler else self.data_handler  # Use passed data_handler if available

        while True:
            try:
                packet = self.conn.recv(4096)
                if not packet:
                    break
                data_buffer += packet

                while b'::' in data_buffer and payload_size is None:
                    header, data_buffer = data_buffer.split(b'::', 1)
                    if header == b'command':
                        if data_buffer == b'quit':
                            self.conn.sendall(b'ack')
                            self.close_connection()
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
                                    self.conn.sendall(encoded_response)
                                else:
                                    self.conn.sendall(b'ack')
                            except Exception as e:
                                tb = traceback.format_exc()
                                logger.error(f"An error occurred in data_handler: {e}\n{tb}")

                        data_buffer = payload_data[payload_size:]
                        payload_data = b""
                        payload_size = None

            except ConnectionResetError:
                logger.error("Connection was closed by the client.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                break

        # Clean up and close the connection
        self.close_connection()

    def close_connection(self):
        logger.info("Closing the server")
        self.conn.close()


# Sample data handler function
def print_received_data(data):
    logger.info(f"Received mask: {data}")

# if __name__ == '__main__':
#     # Initialize SocketServer
#     server = SocketServer(port=5050, data_handler=print_received_data)
#
#     # Accept a client connection
#     server.start_accepting_clients()
