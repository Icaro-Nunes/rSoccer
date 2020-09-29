'''
#   Environment Communication Structure
#    - Father class that creates the structure to communicate with multples setups of enviroment
#    - To create your wrapper from env to communcation, use inherit from this class! 
'''


import gym
import robosim
import numpy as np
from rc_gym.Entities import Frame
from rc_gym.rsim_vss.render import RCRender

class rSimVSSEnv(gym.Env):
    def __init__(self, field_type: int, n_robots: int):
        self.simulator = robosim.SimulatorVSS(field_type=field_type,
                                              n_robots=n_robots)
        self.field_type:int = field_type
        self.n_robots:int = n_robots
        self.field_params: Dict[str, np.float64] = self.simulator.get_field_params()
        self.frame: Frame = None
        self.view = None
        self.steps = 0

    def step(self, action):
        self.steps += 1
        
        # TODO talvez substituir o get commands por wrappers
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Convert commands to simulator commands format
        sim_commands = self.commands_to_sim_commands(commands)
        # step simulation
        # print(sim_commands)
        self.simulator.step(sim_commands)
        
        # Get status and state from simulator
        state = self.simulator.get_state()
        status = self.simulator.get_status()
        # Update frame with new status and state
        self.frame = Frame()
        self.frame.parse(state, status, self.n_robots)
        
        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        return observation, reward, done, {}

    def reset(self):
        self.steps = 0
        
        # Reset simulator
        self.simulator.reset()
        
        # Get status and state from simulator
        state = self.simulator.get_state()
        status = self.simulator.get_status()
        # Update frame with new status and state
        self.frame = Frame()
        self.frame.parse(state, status, self.n_robots)

        return self._frame_to_observations()
    
    def render(self) -> None:
        '''
        Renders the game depending on 
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        if self.view == None:
            self.view = RCRender(self.n_robots, self.n_robots, self.field_params)
            
        state = self.simulator.get_state()
        ball = (state[0], state[1])
        blues = list()
        for i in range(5, self.n_robots*4+6, 4):
            blues.append((state[i], state[i+1]))
        yellows = list()
        for i in range(self.n_robots*4+5, len(state), 4):
            yellows.append((state[i], state[i+1]))

        self.view.view(ball, blues, yellows)
    
    def commands_to_sim_commands(self, commands):
        sim_commands = np.zeros((self.n_robots * 2, 2), dtype=np.float64)
        
        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots + cmd.id
            else:
                rbt_id = cmd.id
            sim_commands[rbt_id][0] = cmd.v_wheel1 * 100
            sim_commands[rbt_id][1] = cmd.v_wheel2 * 100
        
        return sim_commands

    def _get_commands(self, action):
        '''returns a list of commands of type List[VSSRobot] from type action_space action'''
        raise NotImplementedError

    def _frame_to_observations(self):
        '''returns a type observation_space observation from a type List[VSSRobot] state'''
        raise NotImplementedError

    def _calculate_reward_and_done(self):
        '''returns reward value and done flag from type List[VSSRobot] state'''
        raise NotImplementedError


