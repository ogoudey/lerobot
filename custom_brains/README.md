## Base model
`smolvla_base` is a combination of a VLM and an action expert. It is trained according to these [specs](https://huggingface.co/lerobot/smolvla_base/blob/main/train_config.json).

### Fine-tuning

#### Setup
Cameras are linked up in this pipeline as webcams coming from smartphones. Specifically, to use this method, install **IP Webcam** on all cameras. Then get their IP address and plug them into `webcam1_url = ` in `test.py`.

#### Data collection
Right now (and this should be better), I un-comment the desired function:
```
def main():
    """ A repetoire of useful main functions: """
    #merge_datasets("data/<new_dataset_name>", "data/<existing_dataset_1>",  "data/<existing_dataset_2>")
    #check_episode_stats("data/<dataset>/meta/episodes_stats.jsonl") # Checks for anomalies that I've ran into
    
    # I "outsource" the train script (see below)
    
    #record_dataset("<new_dataset_name>")
    #teleoperate(teleop_config()) # to test pure teleoperation (it's good to practice)
    #test_policy("<absolute_path_to_policy/pretrained_model>")
```
And run:
```
python3 custom_brains/test.py
```


To do...
* Standardize/decide on methods of path locating.
* Open up camera functions.
* Automatically get the correct encodings of videos.

#### Training
Currently, the fine-tuned model is trained with
```
python3 src/lerobot/scripts/train.py  --policy.path=lerobot/smolvla_base  --dataset.repo_id=<dataset_path>   --batch_size=16   --steps=<training steps>   --policy.push_to_hub=false --save_freq <save_frequency> (--resume=true)
```
I use a batch size of 16 to fit training on a 8GB RAM GPU.

(I also modify and use `./re-encode` to change the media type of the videos. (Immediately better ways to do this. AI-generated bash script...))


And to resume training, locate the `train_config` of the previously trained model.
```
python3 src/lerobot/scripts/train.py --dataset.repo_id=/home/olin/Robotics/Projects/LeRobot/lerobot/data/f5   --batch_size=16   --steps=30000 --resume=true --config_path=/home/olin/Robotics/Projects/LeRobot/lerobot/outputs/train/2025-09-02/11-51-54_smolvla/checkpoints/last/pretrained_model/train_config.json --policy.push_to_hub=false
```

### Latest Update
See `outputs/*_evaluation` for some notes based on the [questionnaire](https://olimn.com/questionnaire.html).
