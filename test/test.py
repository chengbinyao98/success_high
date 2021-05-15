from agnet1.main1 import Main1
from test.real_environment import Env
import tensorflow as tf
import numpy as np
from tool import Tools

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    real_env = Env()
    tools = Tools()

    gra = tf.Graph()
    main = Main1(gra)
    # rl = DQN1(gra=gra,
    #           # sess,
    #           s_dim=3,
    #           a_dim=real_env.road_range,
    #           batch_size=128,
    #           gamma=0.99,
    #           lr=0.01,
    #           epsilon=0.1,
    #           replace_target_iter=300)

    while True:
        a = input("input:")

        if a == '1':
            main.train()

        if a == '2':
            main.rl.restore_net()

            r_state = real_env.reset()
            for episodes in range(10000):

                # suss = 0
                # total = 0
                # zongzhou = []

                r_action = []
                for num in range(len(r_state[1])):
                    temp_state = tools.get_list(r_state[1][num])  # 车组中所有车辆状态合成
                    temp = main.rl.real_choose_action(np.array(temp_state))
                    r_action.append(int(real_env.cars_posit[num]) - real_env.road_range / 2 + temp)

                r_next_state, r_reward = real_env.step(r_action)

                su = 0
                to = 0
                for i in range(len(r_state)):
                    su += r_reward[i]
                    to += real_env.beam_slot

                # suss += su
                # total += to
                # zongzhou.append(suss / total)

                print('成功率', su / to)
                r_state = r_next_state

                # plt.sca(ax1)
                # ax1.cla()
                # plt.ylim(0.93, 1.01)
                # my_y_ticks = np.arange(0.93, 1.01, 0.01)
                # plt.yticks(my_y_ticks)
                # plt.plot([i for i in range(len(zongzhou))], zongzhou)
                # plt.pause(env.frame_slot)

