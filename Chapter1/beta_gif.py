import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
plt.style.use('fivethirtyeight')

# 尚未 normolize

# uniform alpha 1, beta 1
# bias dis 0.5, 0.5

# expoentail 
# growth  alpha > beta when alpha = 1
# decay alpha < beta when alpha = 1

# gaussian-like alpha = beta 但 != 1, ex 2, 5, 20
# pick 5, 5, 20, 20

params = [0.5, 1, 2, 3, 5, 20]
x = np.linspace(0, 1, 100)
for i in range(len(params)):
    for j in range(len(params)):
        f, ax = plt.subplots(figsize=(12, 8))
        FILE_PATH_beta_gif = f'../gif/beta_alpha{params[i]}_beta_{params[j]}.png'
        a = params[i]
        b = params[j]
        y = stats.beta(a, b).pdf(x) # the meats
        ax.plot(x, y)
        # ax.plot(0, 0, label="$\\alpha$ = {:3.2f}\n$\\beta$ = {:3.2f}".format(a, b), alpha=0)
        # ax.legend(fontsize=12)
        ax.set_xlabel('$\\theta$', fontsize=14)
        ax.set_ylabel('$p(\\theta)$', fontsize=14)
        ax.set_title("$\\alpha$ = {:3.2f}    $\\beta$ = {:3.2f}".format(a, b))
        # plt.show()
        # save = input("Save the figure? y/n")
        # if save == 'y':
        plt.savefig(FILE_PATH_beta_gif, dpi=50, figsize=(12, 8))