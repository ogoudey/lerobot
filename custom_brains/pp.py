from pynput.keyboard import Listener, Controller, Key
import os

# Force terminal backend
os.environ['PYNPUT_BACKEND'] = 'termios'

def on_press(key):
    print(f"Terminal capture: {key}")

with Listener(on_press=on_press) as listener:
    input("Press Enter to quit...")
