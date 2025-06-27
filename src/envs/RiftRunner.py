import numpy as np
import torch as th
from .multiagentenv import MultiAgentEnv

class RiftRunner(MultiAgentEnv):
    def __init__(self,
        map_name="RiftRunner",
        length=20,
        n_agents=3,
        n_actions=6,
        sight_range=3,
        time_step=0,
        episode_limit=200,
        seed=None):
        self.length = length
        self.n_agents = n_agents
        self.n_actions = n_actions #[0-5] for up down left right rightdown stay
        self.time_step = time_step
        self.episode_limit = episode_limit
        self.sight_range = sight_range

        self._init_grid()


    # def _init_grid(self):
    #     '''
    #     init grid
    #     0 represent road;
    #     1 represent wall;
    #     10 represent goal;
    #     -1 represent out of grid.
    #     '''
    #     self.state = np.zeros((self.length,self.length))
    #     # Place walls
    #     self.state[1:self.length-1,1:self.length-1]=1
    #     # Place middle laner 
    #     for i in range(1,self.length-1):
    #         self.state[i,i]=0
    #     # Place Goal
    #     self.state[self.length-1,self.length-1]=10

    #     # init agents
    #     self.agents = [[0,0] for i in range(self.n_agents)] # record each the location of each agent

    def _init_grid(self):
        '''
        init grid
        0 represent road;
        1 represent wall;
        10 represent goal;
        -1 represent out of grid.
        '''
        self.state = np.zeros((self.length,self.length))
        # Place walls
        self.state[1:self.length,1:self.length]=1
        # Place middle laner 
        for i in range(1,self.length-1):
            self.state[i,i]=0
        # Place Goal
        self.state[self.length-1,self.length-1]=10
        self.state[0,self.length-1]=10
        self.state[self.length-1,0]=10

        # init agents
        self.agents = [[0,0] for i in range(self.n_agents)] # record each the location of each agent


    def step(self, actions):
        """ Returns reward, terminated, info """
        if th.is_tensor(actions):
            actions = actions.cpu().numpy().tolist()
        
        done = False
        self.time_step += 1

        cur_positions = []
        for agent_position,action in zip(self.agents,actions):
            x,y = agent_position
            if action == 0: # move up
                #print("move up")
                target_x = x - 1
                target_y = y
                if target_x < 0: # out of range
                    target_x = x
                elif self.state[target_x,y] == 1: # wall
                    target_x = x
            elif action == 1: # move down
                #print("move down")
                target_x = x + 1
                target_y = y
                if target_x >= self.length: # out of range
                    target_x = x
                elif self.state[target_x,y] == 1: # wall
                    target_x = x
            elif action == 2: # move left
                #print("move left")
                target_x = x
                target_y = y - 1
                if target_y < 0: # out of range
                    target_y = y
                elif self.state[x,target_y] == 1: # wall
                    target_y = y
            elif action == 3: # move right
                #print("move right")
                target_x = x
                target_y = y + 1
                if target_y >= self.length: # out of range
                    target_y = y
                elif self.state[x,target_y] == 1: # wall
                    target_y = y
            elif action == 4: # move right down
                #print("move right down")
                target_x = x + 1
                target_y = y + 1
                if target_x >= self.length or target_y >= self.length: # out of range
                    target_x = x
                    target_y = y             
                elif self.state[target_x,target_y] == 1: # wall
                    target_x = x
                    target_y = y
            elif action == 5: # stay
                #print("stay")
                target_x = x
                target_y = y
            else:
                pass
            cur_positions.append([target_x,target_y])
        # update agents position
        for i in range(self.n_agents):
            self.agents[i] = cur_positions[i]
        # coumpute reward
        # reward = 1 when all agent stay on Goal
        for _position in cur_positions:
            x,y = _position
            if self.state[x,y] == 10:
                reward = 1
                continue
            else:
                reward = 0
                break

        if self.time_step >= self.episode_limit or reward == 1:
            done = True
        infos = {}

        return reward, done, infos
        

    def get_obs(self):
        """ Returns all agent observations in a list """
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs = [] # init
        agent_position = self.agents[agent_id]
        x,y = agent_position
        for i in range(-self.sight_range,self.sight_range+1):
            for j in range(-self.sight_range,self.sight_range+1):
                if (x + i > 0) and (x + i < self.length) and (y + j > 0) and (y + j < self.length):
                    obs.append(self.state[x+i,y+j])
                else:
                    obs.append(-1) # out of grid
        return obs



    def get_obs_size(self):
        """ Returns the shape of the observation """
        return (self.sight_range * 2 + 1)**2
    
    def get_state(self):
        states = []
        states += self.state.flatten().tolist()
        for s in self.agents:
            states += s 
        return states
    def get_state_size(self):
        return len(self.get_state())


    def get_avail_actions(self):
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self._init_grid()
        self.time_step = 0

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return  {}
