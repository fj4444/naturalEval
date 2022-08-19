import matplotlib.pyplot as plt
import numpy as np
import ipdb
count_list = []
loss_list = []
prob_list = []
count = 0
with open("./logs/epoch-8+/loss.txt", "r") as f:
    for line in f.readlines():
        splited_line = line.strip("\n").split(",")
        loss = splited_line[0].strip(" ").split(" ")[1].strip(" ")
        prob = splited_line[1].strip(" ").split(" ")[1].strip(" ")
        if loss == "nan":
            pass
        else:
            count_list.append(count)
            if eval(loss) < 5:
                loss_list.append(eval(loss))
            else:
                loss_list.append(0)

            prob_list.append(eval(prob))
        count = count + 1
    count_list = np.array(count_list)
    loss_list = np.array(loss_list)
    prob_list = np.array(prob_list)
    # ipdb.set_trace()
    plt.plot(count_list, loss_list)
    plt.savefig("loss_plot.png")
    plt.cla()
    # ipdb.set_trace()
    plt.plot(count_list, prob_list)
    plt.savefig("prob_plot.png")
    plt.cla()
