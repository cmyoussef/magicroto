import socket
import threading
import time

import nuke

from magicroto.utils.logger import logger
from magicroto.utils.soketserver import SocketServer


class SocketClient:

    def __init__(self, port, set_status, on_receive):
        self.is_client_connected = None
        self.port = port
        self.set_status = set_status
        self.on_receive = on_receive
        self.thread_list = []

    def is_server_running(self):
        """Check if the server is running by attempting to connect to the server's port."""
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            test_socket.connect(("localhost", self.main_port))
            test_socket.close()
            return True
        except socket.error:
            return False

    def connect_to_server(self):
        try:
            if self.mask_client is not None:
                self.mask_client.close()

            self.mask_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.mask_client.connect(("localhost", self.main_port))
            self.mask_client.setblocking(False)
            self.is_client_connected = True
            logger.info(f"Successfully connected to server at port {self.main_port}")
            self.set_status(True, f"Still initializing\nSuccessfully connected to server at port {self.main_port}")
            # self.set_status(True, f"Successfully connected to server at port {self.main_port}")
            return True

        except ConnectionRefusedError:
            logger.warning(f"Connection to server at port {self.main_port} refused.")
            if self.mask_client is not None:
                self.mask_client.close()
            self.is_client_connected = False
            return False

        except Exception as e:
            logger.error(f"Error while connecting: {e}")
            if self.mask_client is not None:
                self.mask_client.close()
            self.is_client_connected = False
            return False

    def attempt_reconnect(self):
        thread = threading.Thread(target=self._attempt_reconnect, args=())
        thread.start()
        logger.debug('Attempt reconnect on another thread')
        self.thread_list.append(thread)

    def _attempt_reconnect(self):
        retry_count = 0
        while not self.connect_to_server() and retry_count < 5:
            time.sleep(1)  # Waiting for 1 second before retrying
            retry_count += 1
            logger.info(f"Retrying connection to server (Attempt {retry_count})")
            if retry_count == 4:
                self.main_port = SocketServer.find_available_port()
        if retry_count == 5:
            logger.error("Failed to connect to the server after multiple attempts.")

    def close_server(self):

        logger.debug(f'attempt to close server at port {self.main_port}, from {self.mask_client}')
        if self.mask_client:
            header = b'command::'
            self.mask_client.sendall(header + b'quit')
            self.mask_client.close()
            self.mask_client = None
            logger.info("Closing Command sent.")
