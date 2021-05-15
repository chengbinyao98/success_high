import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from tool import Tools


class Env(object):
    def __init__(self):
        self.tool = Tools()
        # 帧结构
        self.frame_slot = 0.01  # 帧时隙时间长度
        self.beam_slot = 100  # 波束选择时隙数
        self.right = 5  # 正确传输最低的SNR

        # 车辆和道路
        self.road_length = 200  # 道路长度
        self.straight = 100  # 基站和道路的直线距离

        self.no_interference = 30  # 车辆没有干扰的距离

        self.v_min = 8  # 车辆的最小速度
        self.v_max = 16  # 车辆的最大速度
        self.accelerate = 16  # 车辆的加速度

        self.min_dis = 22  # 车辆之间的最小反应距离

        self.per_section = 5  # 每几米划分成一个路段
        self.road_range = 40  # 动作可以选择的范围

        self.l_min = 22  # 车辆间距服从均匀分布
        self.l_max = 40

        # 天线
        self.ann_num = 16  # 天线数目

        # 存储单元
        self.cars_posit = []  # 车辆的位置（连续）
        self.cars_speed = []  # 车辆的速度（连续)

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, list):
        section = []
        for i in range(len(list)):
            section.append(math.ceil(list[i] / self.per_section))
        return section

    # 道路初始化
    def road_reset(self):
        for i in range(50):  # 任意数目都可以，主要是用于生成路段上的车辆
            # 初始化车辆速度
            speed = np.random.uniform(self.v_min, self.v_max)
            # 初始化车辆间距，使得初始化的车辆间距大于车辆之间开始反应的距离
            dis = np.random.uniform(self.l_min, self.l_max)
            # 生成车辆的初始位置和速度
            if i == 0:
                self.cars_posit.append(dis)
                self.cars_speed.append(speed)
            else:
                y = self.cars_posit[i - 1] + dis
                if y >= self.road_length:
                    break
                else:
                    self.cars_posit.append(y)
                    self.cars_speed.append(speed)

    # 道路路面现有车辆更新
    def road_step(self):
        mark = 0  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
        for i in range(len(self.cars_posit) - 1):
            if mark == 0:
                if self.cars_posit[i + 1] - self.cars_posit[i] < self.min_dis:
                    if np.random.rand() < 0.5:
                        cars_speed_next = self.cars_speed[i] - self.accelerate * self.frame_slot
                        # 减速到最小速度即可
                        if cars_speed_next <= self.v_min:
                            cars_speed_next = self.v_min
                        ti = (self.cars_speed[i] - cars_speed_next) / self.accelerate
                        self.cars_posit[i] = self.cars_speed[i] * ti - ti * ti * self.accelerate / 2 + (
                                self.frame_slot - ti) * cars_speed_next + self.cars_posit[i]
                        self.cars_speed[i] = cars_speed_next
                        mark = 0
                    else:
                        cars_speed_next = self.cars_speed[i + 1] + self.accelerate * self.frame_slot
                        # 减速到最小速度即可
                        if cars_speed_next >= self.v_max:
                            cars_speed_next = self.v_max
                        ti1 = (cars_speed_next - self.cars_speed[i + 1]) / self.accelerate
                        self.cars_posit[i + 1] = self.cars_speed[i + 1] * ti1 + ti1 * ti1 * self.accelerate / 2 + (
                                self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i + 1]
                        self.cars_speed[i + 1] = cars_speed_next
                        self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
                        mark = 1
                else:
                    self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
                    mark = 0
            else:
                if self.cars_posit[i + 1] - self.cars_posit[i] < self.min_dis:
                    cars_speed_next = self.cars_speed[i + 1] + self.accelerate * self.frame_slot
                    # 减速到最小速度即可
                    if cars_speed_next >= self.v_max:
                        cars_speed_next = self.v_max
                    ti1 = (cars_speed_next - self.cars_speed[i + 1]) / self.accelerate
                    self.cars_posit[i + 1] = self.cars_speed[i + 1] * ti1 + ti1 * ti1 * self.accelerate / 2 + (
                            self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i + 1]
                    self.cars_speed[i + 1] = cars_speed_next
                    mark = 1
                else:
                    mark = 0
        if mark == 0:
            self.cars_posit[len(self.cars_posit) - 1] = self.cars_speed[len(self.cars_posit) - 1] * self.frame_slot + \
                                                        self.cars_posit[len(self.cars_posit) - 1]


    def get_information(self, action, section):
        for i in range(10):  # 这个10随便，只要保证能新加上所有的车辆即可
            # 生成一个新的车辆进入，初始化车辆间距
            dis1 = np.random.uniform(self.l_min, self.l_max)
            dis2 = np.random.uniform(self.l_min, self.l_max)
            if self.cars_posit[0] >= dis1 + dis2:
                action.insert(0, (self.cars_posit[0] - dis1) / self.per_section)
                section.insert(0, (self.cars_posit[0] - dis1) / self.per_section)
                self.cars_posit.insert(0, (self.cars_posit[0] - dis1))  # 车辆的位置（位置更新）
                self.cars_speed.insert(0, np.random.uniform(self.v_min, self.v_max))  # 车辆的速度（位置更新）
            else:
                break
        for i in range(10):
            # 将超出道路的车辆排除
            if self.cars_posit[len(self.cars_posit) - 1] > self.road_length:
                del action[len(self.cars_posit) - 1]
                del section[len(self.cars_posit) - 1]
                del self.cars_speed[len(self.cars_posit) - 1]
                del self.cars_posit[len(self.cars_posit) - 1]
            else:
                break
        return action, section

    def get_reward(self, list_act, list_reward):
        info = self.tool.get_info(self.cars_posit, self.no_interference)
        for i in range(len(list_act)):
            SNR_noise = 0
            SNR = 0
            for num in range(i - info[i][2], i - info[i][2] + info[i][0]):
                # 直角边
                a = abs(self.road_length / 2 - self.cars_posit[num])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(self.straight))
                if self.cars_posit[num] > self.road_length / 2:
                    th1 = math.pi - math.acos(a / b)
                else:
                    th1 = math.acos(a / b)

                channel = []
                for t in range(self.ann_num):
                    m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                    channel.append(m.conjugate())

                # 直角边
                c = abs(self.road_length / 2 - list_act[i] )
                # 斜边
                d = np.sqrt(np.square(c) + np.square(self.straight))
                if list_act[i]  > self.road_length / 2:
                    th2 = math.pi - math.acos(c / d)
                else:
                    th2 = math.acos(c / d)

                signal = []
                for t in range(self.ann_num):
                    n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                    signal.append(n)

                if num != i:
                    SNR_noise += np.square(np.linalg.norm(np.dot(channel, signal)))
                else:
                    SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
            if SNR_noise == 0:
                if SNR >= self.right:
                    list_reward[i] += 1
            else:
                if SNR / SNR_noise >= self.right:
                    list_reward[i] += 1
        return list_reward

    def reset(self):
        # 道路环境初始化
        self.road_reset()

        # 获得道路上的每个车辆信息
        info = self.tool.get_info(self.cars_posit, self.no_interference)

        number = [i for i in range(len(self.cars_posit))]

        # 形成状态
        a = self.tool.classify(number, info)
        b = self.tool.classify(self.get_section(self.cars_posit), info)
        d = self.tool.integrate(b, b, b, a)
        return d

    def step(self, list_action):
        info = self.tool.get_info(self.cars_posit, self.no_interference)  # 当前道路的信息
        # print(info)
        # print(tool.get_info1(self.cars_posit, self.no_interference))
        # action = tool.reverse_classify(dic_action, info)  # 当前车辆的动作
        section = self.get_section(self.cars_posit)  # 当前车辆的路段  用于产生下一时刻的状态

        # 道路的（位置更新）
        reward = [0 for p in range(len(info))]  # 用于记录一个帧周期的车辆情况
        for i in range(self.beam_slot):
            self.road_step()
            reward = self.get_reward(list_action, reward)
        # dic_reward = self.tool.classify(reward, info)

        action, change_next_section = self.get_information(list_action, section)

        next_info = self.tool.get_info(self.cars_posit, self.no_interference)  # 下一时刻的道路信息


        # 下一时刻的状态（位置更新）（数目更新）
        a = self.tool.classify(action, next_info)
        b = self.tool.classify(self.get_section(self.cars_posit), next_info)
        c = self.tool.classify(change_next_section, next_info)
        d = self.tool.classify([i for i in range(len(self.cars_posit))],next_info)
        dic_state_ = self.tool.integrate(a, b, c, d)

        return dic_state_, reward

    def draw(self):

        plt.ion()  # 开启交互模式
        plt.figure(figsize=(100, 3))  # 设置画布大小

        # 数据
        y = []
        for i in range(len(self.cars_posit)):
            y.append(0)

        for j in range(1000):
            plt.clf()  # 清空画布
            plt.axis([0, 210, 0, 0.1])  # 坐标轴范围
            x_major_locator = MultipleLocator(5)  # 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()  # ax为两条坐标轴的实例
            ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
            plt.tick_params(axis='both', which='major', labelsize=5)  # 坐标轴字体大小

            self.road_step()

            plt.scatter(self.cars_posit, y, marker="o")  # 画图数据
            plt.pause(0.2)

        plt.ioff()
        plt.show()

