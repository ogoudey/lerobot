### Base model
`smolvla_base` is a combination of a VLM and an "action expert". It is trained according to the [specs](https://huggingface.co/lerobot/smolvla_base/blob/main/train_config.json).

### Fine-tuning (outdated)
The custom [dataset](https://huggingface.co/datasets/olingoudey/so101puttingcubeinbowl) trains the fine tuned model (specs to come).

Currently, the fine-tuned model is trained with
```
python3 src/lerobot/scripts/train.py  --policy.path=lerobot/smolvla_base  --dataset.repo_id=/home/olin/Robotics/Projects/LeRobot/lerobot/data/f5   --batch_size=16   --steps=1000   --policy.push_to_hub=false --save_freq 100
```
(I also modify and use `./re-encode` to change the media type of the videos. (Immediately better ways to do this. AI-generated bash script...))


And to resume training, locate the `train_config` of the previously trained model.
```
python3 src/lerobot/scripts/train.py --dataset.repo_id=/home/olin/Robotics/Projects/LeRobot/lerobot/data/f5   --batch_size=16   --steps=30000 --resume=true --config_path=/home/olin/Robotics/Projects/LeRobot/lerobot/outputs/train/2025-09-02/11-51-54_smolvla/checkpoints/last/pretrained_model/train_config.json --policy.push_to_hub=false
```

### Latest Update
Convergence! SO-101 picks up the stuffed animal and puts it in bowl...
