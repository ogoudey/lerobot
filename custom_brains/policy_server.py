import sys
import threading
import socket
import time
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import json
from lerobot.utils.control_utils import predict_action
import struct
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="normal",
    use_degrees=False,
)
robot = SO101Follower(robot_config) # not true, but predict action needs it...


print("Loading policy")

device = "cuda"
policy = SmolVLAPolicy.from_pretrained(sys.argv[1])

_infer_lock = threading.Lock()


def recv_exact(sock, n):
    """Read exactly n bytes from sock, blocking until done."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes received")
        buf += chunk
    return buf


def send_msg(sock, payload: bytes):
    """Send a length-prefixed message."""
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_msg(sock) -> bytes:
    """Receive a length-prefixed message."""
    raw_len = recv_exact(sock, 4)
    msg_len = struct.unpack("!I", raw_len)[0]
    return recv_exact(sock, msg_len)

def decode_observation(payload: bytes) -> tuple:
    """
    Payload layout:
      [4 bytes: header_len][header JSON][image1 bytes][image2 bytes][state bytes]
    Header JSON contains shapes, dtypes, and task string.
    """
    offset = 0
    header_len = struct.unpack("!I", payload[offset:offset+4])[0]
    offset += 4
    header = json.loads(payload[offset:offset+header_len])
    offset += header_len

    # Decode image 1
    img1_bytes = header["img1_bytes"]
    img1 = np.frombuffer(payload[offset:offset+img1_bytes], dtype=header["img1_dtype"])\
             .reshape(header["img1_shape"])
    offset += img1_bytes

    # Decode image 2
    img2_bytes = header["img2_bytes"]
    img2 = np.frombuffer(payload[offset:offset+img2_bytes], dtype=header["img2_dtype"])\
             .reshape(header["img2_shape"])
    offset += img2_bytes

    # Decode state
    state_bytes = header["state_bytes"]
    state = np.frombuffer(payload[offset:offset+state_bytes], dtype=header["state_dtype"])\
              .reshape(header["state_shape"])

    observation = {
        "image1": img1,
        "image2": img2,
        "state": state,
    }
    return observation, header["task"]

def handle_client(sock, addr):
    try:
        while True:
            # 1. Receive one full observation message
            try:
                payload = recv_msg(sock)
            except ConnectionError:
                print(f"Client disconnected: {addr}")
                break

            # 2. Decode
            observation, task = decode_observation(payload)

            # 3. Run inference (serialized across clients)
            with _infer_lock:
                action_values = predict_action(
                    observation,
                    policy,
                    device="cuda",
                    use_amp=True,
                    task=task,
                    robot_type=robot.robot_type,
                )

            # 4. Send action back as JSON
            # action_values is likely a numpy array or tensor — convert to list
            if hasattr(action_values, "tolist"):
                action_list = action_values.tolist()
            else:
                action_list = action_values
            response = json.dumps({"action": action_list}).encode()
            send_msg(sock, response)

    finally:
        sock.close()



def run_server():
    print("Opening socket...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 5006))
    server.listen()
    print("Chat server waiting...")
    listening = True
    while listening:
        client_sock, addr = server.accept()
        print("Client connected:", addr)
        threading.Thread(
            target=handle_client,
            args=(client_sock,),
            daemon=True
        ).start()


if __name__ == "__main__":
    
    # create session
    
    print("Starting server loop")
    while True:
        threading.Thread(target=run_server, daemon=True).start()