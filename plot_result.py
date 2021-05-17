import matplotlib.pylab as plt
from matplotlib.pyplot import MultipleLocator

# 绘制车辆间距的变换
x1 = [1, 2, 3, 4, 8]
y1 = [5, 7, 1, 5, 2]

# pyl.title("")
plt.xlabel("车辆间距", fontsize = 14)
plt.ylabel("成功率", fontsize=14)

x_major_locator = MultipleLocator(1)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(10)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10的倍数

plt.xlim(-0.5, 11)
# 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(-5, 110)
# 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
plt.show()

