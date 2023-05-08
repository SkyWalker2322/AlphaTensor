# Requirements
Make sure you have the following libraries
- pytorch               Version: 2.0.0+cu117
- Tensorboard           Version: 2.12.0
- pytorch-lightning     Version: 1.7.1
- tqdm                  Version: 4.65.0
- jupyter               version==1.0.0
- python                version>= 3.10.8

# Usage
create a logs directory of preferred name{logs_1} in the logs directory, and in ```main.py``` change ln:28 to the newly created logs ex:{logs/logs_1}. then we are finally ready to go. Now execute the following command.
```bash
python main.py
```
To View the Results 
```bash
tensorboard --logdir ./logs/logs_1
```

# Furthur Optimizations
- Policy Improvement in MCTS
- 