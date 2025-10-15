import requests

# lerobot imports, import get_action_chunk()


@app.route('/get_action_chunk')
def get_action_chunk():
    observation_frame = requests.get_data("observation_frame")
    task = requests.get_data("task")
    
    action_values = predict_action(
                        observation_frame,
                        smolvla_policy,
                        device=device,
                        use_amp=(device.type == "cuda"),
                        task=task,
                        robot_type=robot,
                    )
    return action_chunk
    
def main():
    # get policy -> policy

    # init robot

    while True:
        
        
    
if __name__ == "__main__":
    main()

