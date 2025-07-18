import socket


class UdpServer:
    def __init__(self, server_address='localhost', server_port=6340):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.server_address = server_address
        self.server_port = server_port
        self.client_address = None

        self.server_socket.bind((self.server_address, self.server_port))
        print("UDP Server is running!")

    def disconnect(self):
        self.server_socket.close()
        print("Disconnection Successful")

    def receive_msg(self):
        msg, self.client_address = self.server_socket.recvfrom(1024)

        return msg

    def send_msg(self, message, client_address='localhost', port=6340):
        self.server_socket.sendto(message, self.client_address)


class UdpClient:
    def __init__(self, server_address='localhost', server_port=6340):
        self.client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.addr = server_address
        self.port = server_port
        self.client_socket.connect((self.addr, self.port))
        print("UDP Client is Running!")

    def disconnect(self):
        self.client_socket.close()
        print("Disconnection Successful")

    def send_msg(self, message):
        self.client_socket.sendto(message, (self.addr, self.port))

    def receive_msg(self):
        return self.client_socket.recvfrom(1024)[0]
