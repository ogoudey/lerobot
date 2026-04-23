import socket
import struct
import numpy as np
import json

SERVER_HOST = "192.168.idk.idk"
SERVER_PORT = 5006



def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes received")
        buf += chunk
    return buf

def send_msg(sock, payload: bytes):
    sock.sendall(struct.pack("!I", len(payload)) + payload)

def recv_msg(sock) -> bytes:
    raw_len = recv_exact(sock, 4)
    msg_len = struct.unpack("!I", raw_len)[0]
    return recv_exact(sock, msg_len)

_sock = None

def _get_socket():
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.connect((SERVER_HOST, SERVER_PORT))
        print(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")
    return _sock

def encode_observation(observation_frame: dict, task: str) -> bytes:
    """
    observation_frame expected keys:
      "image1": np.ndarray  (H, W, 3) uint8
      "image2": np.ndarray  (H, W, 3) uint8
      "state":  np.ndarray  (D,)      float32

    Layout: [4B header_len][header JSON][img1 raw][img2 raw][state raw]
    """
    img1: np.ndarray = observation_frame["image1"]
    img2: np.ndarray = observation_frame["image2"]
    state: np.ndarray = observation_frame["state"]

    img1_raw = img1.tobytes()
    img2_raw = img2.tobytes()
    state_raw = state.tobytes()

    header = {
        "task": task,
        "img1_shape": list(img1.shape),
        "img1_dtype": str(img1.dtype),
        "img1_bytes": len(img1_raw),
        "img2_shape": list(img2.shape),
        "img2_dtype": str(img2.dtype),
        "img2_bytes": len(img2_raw),
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "state_bytes": len(state_raw),
    }
    header_bytes = json.dumps(header).encode()
    header_len = struct.pack("!I", len(header_bytes))

    return header_len + header_bytes + img1_raw + img2_raw + state_raw

def request_and_wait(observation_frame: dict, task: str) -> list:
    """
    Send an observation frame + task to the GPU server.
    Blocks until the action response arrives.
    Returns action as a Python list of floats.
    """
    sock = _get_socket()
    payload = encode_observation(observation_frame, task)
    send_msg(sock, payload)
    response_bytes = recv_msg(sock)
    response = json.loads(response_bytes)
    return response["action"]


# ---------- example usage ----------
if __name__ == "__main__":
    observation_frame = {
        "image1": np.zeros((480, 640, 3), dtype=np.uint8),
        "image2": np.zeros((480, 640, 3), dtype=np.uint8),
        "state":  np.zeros((6,),        dtype=np.float32),
    }
    task = "pick up the red block"

    action_values = request_and_wait(observation_frame, task)
    print("Received action:", action_values)


