
from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# 指定输出的 CSV 文件路径
output_csv_file = 'C:/Users/PU/Desktop/paper data/agent5122222val_loss.csv'

# 打开 CSV 文件以写入数据
# with open(output_csv_file, 'w', newline='') as csvfile:
    # 定义 CSV 文件的列名
    # fieldnames = ['Step', 'Value']
    #
    # # 创建 CSV writer 对象
    # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # ea = event_accumulator.EventAccumulator('C:/Users/PU/Desktop/paper data')  # 初始化EventAccumulator对象
    # ea.Reload()
    # print(ea.scalars.Keys())
    # # 写入列名
    # writer.writeheader()
    #
    # # 从 event_accumulator 中获取 train_loss 的数据
    # for scalar_event in ea.Scalars("val/mseloss"):
    #     # 获取 step 和 value
    #     step = scalar_event.step
    #     value = scalar_event.value
    #
    #     # 将数据写入 CSV 文件
    #     writer.writerow({'Step': step, 'Value': value})

df = pd.read_csv('C:/Users/PU/Desktop/paper data/ATL.csv')

number = df['number']
parameters = df['parameters']
runtime = df['runtime']
psnr = df['psnr']
ssim = df['ssim']
rmse = df['rmse']

fig, ax1 = plt.subplots(figsize=(10, 6))
# 绘制 parameters 对应的折线图（使用左侧纵坐标轴）
ax1.plot(number, rmse, marker='o', markersize=8, linestyle='-', color='red', label='RMSE', markerfacecolor='w', markeredgewidth=1)
ax1.set_xlabel('number of A-SWTL',fontsize=25)  # 设置横坐标轴标签
ax1.set_ylabel('RMSE', color='red',fontsize=25)  # 设置左侧纵坐标轴标签颜色
ax1.tick_params(axis='y', labelcolor='red',labelsize = 20)  # 设置左侧纵坐标轴刻度颜色
ax1.tick_params(axis='x', labelsize=20)

#创建第二个纵坐标轴对象，并绘制 throughput 对应的折线图
ax2 = ax1.twinx()  # 创建第二个纵坐标轴（共享横坐标轴）
ax2.plot(number,psnr, marker='s', markersize=8, linestyle='-', color='orange', label='PSNR')
ax2.set_ylabel('PSNR', color='orange',fontsize=25)  # 设置右侧纵坐标轴标签颜色
ax2.tick_params(axis='y', labelcolor='orange',labelsize = 20)  # 设置右侧纵坐标轴刻度颜色
#ax1.set_ylim(12, 120)
ax2.set_ylim(33.60,33.72)
# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right',fontsize=16)  # 合并图例并放置在最佳位置
#ax1.legend(lines , labels , loc='best',fontsize=20)
# 显示图形
plt.show()

# y_min, y_max = plt.ylim()  # 获取当前 y 轴范围
# x_position = 10
# length = y_max -y_min
# line_length = 0.02  # 控制竖线长度
#
# print(y_min)
#
# # 计算箭头连接线的终点位置
# x_end = x_position
# y_end = y_min + line_length
# std_dev = 0.02
#
#
# plt.title('')
# plt.xlabel('Epoch',fontsize = 18)
# plt.ylabel('MSE Loss',fontsize = 18)
# plt.grid(False)  # 不显示网格线
# plt.legend(fontsize='x-large')
#
# ax = plt.gca()
# ax.tick_params(axis='y', labelsize=18)
# ax.tick_params(axis='x', labelsize=18)
# plt.show()


