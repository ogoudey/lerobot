import socket
import json

UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", UDP_PORT))

print("Listening...")

while True:
    data, addr = sock.recvfrom(4096)
    msg = json.loads(data.decode())
    print("Received:", msg)
