from agnet1.environment1 import Env1
from agnet1.dqn1 import *
import matplotlib.pyplot as plt
import numpy as np
from agnet1.draw1 import DRAW

class Main1(object):
    def __init__(self,g):
        self.env = Env1()
        self.draw = DRAW()
        self.rl = DQN1(
                gra = g,
                # sess,
                s_dim=3,
                a_dim=self.env.road_range,
                batch_size=128,
                gamma=0.99,
                lr=0.01,
                epsilon=0.1,
                replace_target_iter=300
                )

    def train(self):
        plt.ion()
        plt.figure(figsize=(100, 5))  # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # reward图
        epi = []
        success = []

        for episode in range(1200):
            print('episode', episode)
            epi.append(episode)

            state = self.env.reset()

            total_reward = 0
            time = 0

            while True:
                add_action = self.rl.choose_action(np.array(state))  # 学习到车组的动作组合
                action = int(self.env.cars_posit) - self.env.road_range / 2 + add_action

                # self.draw.piant(self.env.cars_posit, self.env.road_range, ax1, self.env.frame_slot, action)

                state_, reward, done = self.env.step(action, state)  # dicreward改成一个值

                self.rl.store_transition_and_learn(state, add_action, reward, state_, done)

                state = state_
                total_reward += reward
                time += 1
                if done:
                    self.rl.saver_net()
                    break

            plt.sca(ax2)
            ax2.cla()
            plt.ylim(0.6, 1.05)
            success.append(total_reward / (self.env.beam_slot * time))
            plt.plot(epi, success)
            plt.pause(self.env.frame_slot)

    # def restore(self):
    #     self.rl.restore_net()

#
#
# if __name__ == '__main__':
#     main = Main1()
#     main.start()
#     main.train()







