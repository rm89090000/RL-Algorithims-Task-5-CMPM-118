import vista
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import numpy as np
import matplotlib.pyplot as plt


#Vista
#change directory as needed.
#tensor board path
path =  "./vista_combined_logs/"
#write to stdout using tensboard
log = configure(path, ["stdout", "tensorboard"])
directory = "/Users/riddhi/Desktop/aiea/vista_traces/20210726-131322_lexus_devens_center"




vista_world = vista.World([directory])
#create car and make it visible in virtual environment
car = vista_world.spawn_agent({})

#PPO algoritim using Gym module
setup = gym.make("CartPole-v1")
#how much to make it move. Use numpy to create a list of 2 numbers
setup.action_space = gym.spaces.Box(-1, 1, (2,))
#use MLP policy
RL = PPO("MlpPolicy", setup)
RL.set_logger(log)

vista_world.reset()
count = 0
print("Printing Data")

#values
val = []
for i in range(200):
    values = np.array([car.relative_state.x, car.relative_state.yaw, car.speed,car.ego_dynamics.steering])
    #predict movement
    x, y = RL.predict(values)
    car.step_dynamics(x.flatten())

    #use the reward formula reward = 1- |X|
    reward = 1 - abs(car.relative_state.x)
    #add values to the list
    val.append(reward)

    count+=1
    if count >= 100:
        print(f"Relative State x: {car.relative_state.x,}, Relative state yaw: {car.relative_state.yaw},Speed:{car.speed}, Steering: {car.ego_dynamics.steering}")
        count = 0

#use matplotlib to plot data. It can take a long time because there are 200 points.
plt.figure()
plt.plot(val)
plt.title("Time vs. Rewards")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.show()
vista_world.reset()


