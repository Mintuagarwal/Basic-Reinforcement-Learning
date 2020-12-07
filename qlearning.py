import gym
import numpy as np

env = gym.make("MountainCar-v0")
#env.reset()

alpha = 0.18
discount = 0.95
Episodes = 500
freq = 100

discrete_os_size = [20] * len(env.observation_space.low)
discrete_block_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size

q_values = np.random.normal(0, 2, size = (discrete_os_size + [env.action_space.n]))

def get_discrete_state(state):
    dis_state = (state - env.observation_space.low)/discrete_block_size
    dis_state = dis_state.astype(np.int)
    return tuple(dis_state)

#dis_state = get_discrete_state(env.reset())
#print(q_values[dis_state].shape)

for episode in range(Episodes + 1) :
    done = False
    dis_state = get_discrete_state(env.reset())
    cnt = 0
    if episode % freq == 0:
        print(q_values)
        print(episode)
    while not done:
        #print(dis_state)
        action = np.argmax(q_values[dis_state])
        new_state, reward, done, valx = env.step(action)
        temp_dis_state = get_discrete_state(new_state)
        if episode % freq == 0 :
            env.render()
        if not done:
            curr = q_values[dis_state + (action, )]
            mx = np.max(q_values[temp_dis_state])
            q_values[dis_state + (action, )] = (1-alpha)*curr + alpha*(reward + discount*mx)
        elif new_state[0] >= env.goal_position :
            q_values[dis_state + (action, )] = 0
        dis_state = temp_dis_state
        cnt += 1
        if cnt > 1000 :
            break
        #print(cnt)
                                                                 
env.close()