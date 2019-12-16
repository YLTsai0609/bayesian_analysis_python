import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
plt.style.use('fivethirtyeight')

# 尚未 normolize

# uniform alpha 1, beta 1
# bias dis 0.5, 0.5

# expoentail 
# growth alpha > beta when alpha = 1
# decay alpha < beta when alpha = 1

# gaussian-like alpha = beta 但 != 1, ex 2, 5, 20
# pick 5, 5, 20, 20

# 這裡指的Gaussian-like並不是說真的是高斯函數，而是長得像高斯函數，例如alpha=beta=2的圖型
# 並沒有反曲點，所以也不能稱作Gaussian函數，但我們可以看到 alpha=beta=3, 5, 20時，我們確實可以看到反曲點
def get_subtitle(alpha : int, beta : int) -> str:
    if alpha == beta:
        if alpha == beta == 0.5:
            return '$biasd~~~distrubtion$'
        elif alpha == beta == 1:
            return '$uniform~~~distribution$'
        else:
            return '$gaussian-like~~~distribution$'
    elif alpha > beta:
        if beta == 0.5 or beta == 1:
            return '$growth-like~~~distribution$'
        else:
            return '$negtive~~skew~~gaussian-like~~~distribution$'
    elif alpha < beta:
        if alpha == 0.5 or alpha == 1:
            return '$decay-like~~~distribution$'
        else:
            return '$positve~~skew~~gaussian-like~~~distribution$'


# ref 
# https://matplotlib.org/3.1.1/gallery/statistics/histogram_features.html#sphx-glr-gallery-statistics-histogram-features-py
# idx = 0
# for Path, Directory, files in os.walk("C:\\照片路徑"):
#     for sfile in files:
#         Serial_Num = 0
#         NewFileName = "000_%s.txt" % str(Serial_Num)

def get_prefix(idx):
    if idx < 10:
        return f'000{idx}'
    elif idx < 100:
        return f'00{idx}'
    else:
        return
idx = 0
params = [0.5, 1, 2, 3, 5, 20]
x = np.linspace(0, 1, 100)
for i in range(len(params)):
    for j in range(len(params)):
        f, ax = plt.subplots(figsize=(12, 8))
        prefix = get_prefix(idx)
        FILE_PATH_beta_gif = f'../gif/{prefix}_beta_alpha{params[i]}_beta_{params[j]}.png'
        print(FILE_PATH_beta_gif)
        a = params[i]
        b = params[j]
        y = stats.beta(a, b).pdf(x) # the meats
        ax.plot(x, y)
        ax.set_xlabel('$\\theta$', fontsize=14)
        ax.set_ylabel('$p(\\theta)$', fontsize=14)
        subtitle = get_subtitle(alpha = a, beta = b)
        ax.set_title("$\\alpha$ = {0:3.2f}    $\\beta$ = {1:3.2f}\n{2}".format(a, b,subtitle))
        # plt.show()
        # save = input("Save the figure? y/n")
        # if save == 'y':
        plt.savefig(FILE_PATH_beta_gif, dpi=50, figsize=(12, 8))
        idx += 1
    