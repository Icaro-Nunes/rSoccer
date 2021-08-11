import gym
import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSSMotionTuning-v0')

# Run for 1 episode and print reward at the end
while True:#for i in range(10):
    env.reset()
    done = False
    print("reset")
    while not done:
        # Step using random actions
        action = [0.0, 0.0]#env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        #print(next_state)
        #env.render()
        #print(reward)