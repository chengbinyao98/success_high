import numpy as np
import math


class Env1(object):
    def __init__(self):
        # 帧结构
        self.frame_slot = 0.01          # 帧时隙时间长度
        self.beam_slot = 100       # 波束选择时隙数
        self.right = 5                 # 正确传输最低的SNR
        # self.frame_period = self.beam_slot * self.frame_slot  # 帧周期时间长度

        # 车辆和道路
        self.road_length = 200          # 道路长度
        self.straight = 100             # 基站和道路的直线距离

        self.v_min = 8                  # 车辆的最小速度
        self.v_max = 16                 # 车辆的最大速度
        self.accelerate = 16             # 车辆的加速度

        self.per_section = 5          # 每几米划分成一个路段
        self.road_range = 40                               # 动作可以选择的范围

        # 天线
        self.ann_num = 16                                  # 天线数目

        # 存储单元
        self.cars_posit = 0            # 车辆的位置（连续）
        self.cars_speed = 0            # 车辆的速度（连续

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, pos):
        section = math.ceil(pos / self.per_section)
        return section

    def get_reward(self, act, reward):
        # 直角边
        a = abs(self.road_length / 2 - self.cars_posit)
        # 斜边
        b = np.sqrt(np.square(a) + np.square(self.straight))
        if self.cars_posit > self.road_length / 2:
            th1 = math.pi - math.acos(a / b)
        else:
            th1 = math.acos(a / b)

        channel = []
        for t in range(self.ann_num):
            m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
            channel.append(m.conjugate())

        # 直角边
        c = abs(self.road_length / 2 - act)
        # 斜边
        d = np.sqrt(np.square(c) + np.square(self.straight))
        if act > self.road_length / 2:
            th2 = math.pi - math.acos(c / d)
        else:
            th2 = math.acos(c / d)

        signal = []
        for t in range(self.ann_num):
            n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
            signal.append(n)

        SNR = np.square(np.linalg.norm(np.dot(channel, signal)))

        if SNR >= self.right:
            reward += 1
        return reward

    def reset(self):
        # 道路环境初始化
        self.cars_speed = np.random.uniform(self.v_min, self.v_max)
        self.cars_posit = 0
        # 形成状态
        state = [0,0,0]
        return state

    def step(self, action, state):

        # 道路的（位置更新）
        reward = 0
        for i in range(self.beam_slot):  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
            self.cars_posit += self.cars_speed * self.frame_slot
            reward = self.get_reward(action,reward)

        state_ =[action, self.get_section(self.cars_posit), state[1]]

        if self.cars_posit > self.road_length:
            done = 1
        else:
            done = 0

        return state_, reward, done

