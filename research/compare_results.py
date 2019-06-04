import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with open('./research/results/calmar_net_worths_0.pkl', 'rb') as handle:
    calmar_net_worths_0 = pickle.load(handle)

with open('./research/results/calmar_net_worths_1.pkl', 'rb') as handle:
    calmar_net_worths_1 = pickle.load(handle)

with open('./research/results/calmar_net_worths_2.pkl', 'rb') as handle:
    calmar_net_worths_2 = pickle.load(handle)

# with open('./research/results/calmar_net_worths_3.pkl', 'rb') as handle:
#     calmar_net_worths_3 = pickle.load(handle)

# with open('./research/results/calmar_net_worths_4.pkl', 'rb') as handle:
#     calmar_net_worths_4 = pickle.load(handle)

with open('./research/results/buy_and_hodl_net_worths.pkl', 'rb') as handle:
    buy_and_hodl_net_worths = pickle.load(handle)

with open('./research/results/rsi_divergence_net_worths.pkl', 'rb') as handle:
    rsi_divergence_net_worths = pickle.load(handle)

with open('./research/results/sma_crossover_net_worths.pkl', 'rb') as handle:
    sma_crossover_net_worths = pickle.load(handle)

plt.plot(calmar_net_worths_0, label="Calmar 1", color="#000033")
plt.plot(calmar_net_worths_1, label="Calmar 2", color="#000066")
plt.plot(calmar_net_worths_2, label="Calmar 3", color="#000099")
# plt.plot(calmar_net_worths_3, label="Calmar 4", color="#0000cc")
# plt.plot(calmar_net_worths_4, label="Calmar 5", color="#0000ff")

plt.plot(buy_and_hodl_net_worths, label="Buy and HODL", color="#006600", alpha=0.4)
plt.plot(rsi_divergence_net_worths, label="RSI Divergence", color="#00aa00", alpha=0.4)
plt.plot(sma_crossover_net_worths, label="SMA Crossover", color="#00ff00", alpha=0.4)

profit_0 = calmar_net_worths_0[-1] - calmar_net_worths_0[0]
profit_1 = calmar_net_worths_1[-1] - calmar_net_worths_1[0]
profit_2 = calmar_net_worths_2[-1] - calmar_net_worths_2[0]
# profit_3 = calmar_net_worths_3[-1] - calmar_net_worths_3[0]
# profit_4 = calmar_net_worths_4[-1] - calmar_net_worths_4[0]

average_profit = round(np.mean([profit_0, profit_1, profit_2]), 2)

plt.legend()
plt.title("Calmar-Based Strategy ($" + str(average_profit) + " avg. profit)")

plt.show(block=True)