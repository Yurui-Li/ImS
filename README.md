# ImS
Source code for paper "Improving Stability of Parameter Sharing in Cooperative Multi-Agent Reinforcement Learning"

## Training

To train `ImS` on GRF: 

```shell
python3 src/main.py --config=qmix --env-config=gfootball with env_args.map_name=academy_3_vs_1_with_keeper env_args.num_agents=3 agent=ims mac=ims learner=ims
```

