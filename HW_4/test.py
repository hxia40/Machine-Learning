import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

alpha1_list = []
alpha2_list = []
alpha3_list = []
alpha4_list = []

episode_list = []

range_end = 100000
for episode in range(range_end):
    episode_list.append(episode)
    alpha1 = max(1 - math.log(episode+1, 10) / 5, 0.001)
    alpha2 = max(1 - math.log(episode + 1, 10) / 4, 0.001)
    alpha3 = max(1 - math.log(episode + 1, 10) / 3, 0.001)
    alpha4 = max(1 - math.log(episode+1, 10) / 2, 0.001)

    alpha1_list.append(alpha1)
    alpha2_list.append(alpha2)
    alpha3_list.append(alpha3)
    alpha3_list.append(alpha4)
# plt.plot(episode_list, alpha1_list, label = '10')
# plt.plot(episode_list, alpha1_list, label = '5')
# plt.plot(episode_list, alpha1_list, label = '2')
plt.plot(episode_list, alpha1_list, label = '100')

plt.legend()
plt.savefig('1.png')
plt.gcf().clear()
