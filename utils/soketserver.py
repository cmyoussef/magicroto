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
    def encode_data(cls, data):
        serialized_data = pickle.dumps(data)
        payload_size = struct.pack('!I', len(serialized_data))
        return b'data::' + payload_size + serialized_data

    @classmethod
    def decode_data(cls, received_data):
        payload_size = struct.unpack('!I', received_data[:4])[0]
        actual_data = received_data[4:payload_size + 4]
        return pickle.loads(actual_data)

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

    def start_accepting_clients(self, data_handler=None):
        thread = threading.Thread(target=self.accept_client, args=(data_handler,))
        thread.start()
        logger.debug(f"Starting to accept clients. on {thread}")

    def accept_client(self, data_handler=None):
        logger.debug("Inside accept_client.")
        self.conn, self.addr = self.server_socket.accept()
        logger.debug(f"Accepted connection from {self.addr}.")
        thread = threading.Thread(target=self.receive_data, args=(data_handler,))
        thread.start()

    def receive_data(self, data_handler=None):
        logger.debug("Inside receive_data.")

        data_buffer = b""
        payload_data = b""
        payload_size = None
        handler = data_handler if data_handler else self.data_handler  # Use passed data_handler if available

        while True:
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
                            if handler:
                                handler(decoded_data)
                            # Sending an acknowledgment back to the client
                            self.conn.sendall(b'ack')
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(f"An error occurred in data_handler: {e}\n{tb}")

                    data_buffer = payload_data[payload_size:]
                    payload_data = b""
                    payload_size = None

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
