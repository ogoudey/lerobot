from camera_readers import LogitechReader

def do():
    webcam1_idx = 2
    webcam2_idx = 4
    webcam1_cap = LogitechReader.get_cap(webcam1_idx)
    webcam2_cap = LogitechReader.get_cap(webcam2_idx)
    webcam1_reader = LogitechReader(webcam1_cap)
    webcam2_reader = LogitechReader(webcam2_cap)
    webcam1_reader.start()
    webcam2_reader.start()
    print("Shows")
    input("Doesnt show - fixed: does show")

if __name__ == "__main__":
    do()