# -*- coding: utf-8 -*-
# %matplotlib inline
import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
palette = 'muted'
sns.set_palette(palette); sns.set_color_codes(palette)

print('PyMC3 : ',pm.__version__)
print('numpy : ',np.__version__)
import scipy
print('scipy : ',scipy.__version__)


# # the posterior

def posterior_grid(grid_points=100, heads=6, tosses=9):
    """
    A grid implementation for the coin-flip problem
    """
    # define a grid
    grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform prior

    #prior = (grid  <= 0.4).astype(int)  # truncated
    #prior = abs(grid - 0.5)  # "M" prior

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(heads, tosses, grid) 

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()

    return grid, posterior


# ## brute force

# +
FILE_PATH_brute_force = '../images/brute_force.png'

points = 15
h, n = 1, 4
grid, posterior = posterior_grid(points, h, n)
plt.plot(grid, posterior, 'o-')
plt.plot(0, 0, label='heads = {}\ntosses = {}'.format(h, n), alpha=0)
plt.xlabel(r'$\theta$', fontsize=14)
plt.legend(loc=0, fontsize=14)

save = input("Save the figure? y/n")
if save == 'y':
    plt.savefig(FILE_PATH_brute_force, dpi=300, figsize=(5.5, 5.5));
# -

# ## Monte Carlo

# +
# Monte Carlo
# 對於很多很難計算或是不可計算的問題，我們可以用隨機採樣計算或是模擬來進行研究

FILE_PATH_monte_carlo = '../images/monte_carlo.png'

N = 10000

x, y = np.random.uniform(-1, 1, size=(2, N))
inside = (x**2 + y**2)  <= 1
pi = inside.sum()*4/N
error = abs((pi - np.pi)/pi)* 100

outside = np.invert(inside)

plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label='$\hat \pi$ = {:4.3f}\nerror = {:4.3f}%'.format(pi, error), alpha=0)
plt.axis('square')
plt.legend(frameon=True, framealpha=0.3, fontsize=16);

save = input("Save the figure? y/n")
if save == 'y':
    plt.savefig(FILE_PATH_monte_carlo, dpi=300, figsize=(5.5, 5.5))

    
# -

# ## Metroplis Hasting

# +
# metropolis
def metropolis(func, steps=10000):
    """A very simple Metropolis implementation"""
    samples = np.zeros(steps)
    old_x = func.mean()
    old_prob = func.pdf(old_x)

    for i in range(steps):
        new_x = old_x + np.random.normal(0, 1)
        new_prob = func.pdf(new_x)
        acceptance = new_prob/old_prob
        if acceptance >= np.random.random():
            samples[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            samples[i] = old_x
    return samples


FILE_PATH_metropolis_hasting = '../images/metropolis_hasting.png'
np.random.seed(345)
func = stats.beta(0.4, 2)
samples = metropolis(func=func)
x = np.linspace(0.01, .99, 100)
y = func.pdf(x)
plt.xlim(0, 1)
plt.plot(x, y, 'r-', lw=3, label='True distribution')
plt.hist(samples, bins=30, normed=True, label='Estimated distribution')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$pdf(x)$', fontsize=14)
plt.legend(fontsize=14)

save = input("Save the figure? y/n")
if save == 'y':
    plt.savefig(FILE_PATH_metropolis_hasting, dpi=300, figsize=(5.5, 5.5))
# -
# # PyMC3 for Coin toss


# 二元白努力分佈，real value 取0.35
np.random.seed(123)
n_experiments = 4
theta_real = 0.35  # unkwon value in a real experiment
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments) # rvs means random variable
data


# * Model Description
#     * 定義$\theta$為硬幣偏差，$y$為$N$次實驗中正面朝上的次數
#     * 根據貝氏定理，我們有 : $p(\theta | y) \propto p(y | \theta)p(\theta)$
#     * likelihood : 多次硬幣投擲之間沒有交互影響(獨立)，而且結果只有兩種可能，可以選擇二項分佈
#     * prior distribution 先驗分佈 : 選擇likelihood的共軛先驗(conjugate prior)，與likelihood相乘之後仍然為原分佈形式，二項分布的共軛先驗為beta分佈: $\theta ～ Beta(\alpha, \beta)$

# * 初始參數，$\alpha = \beta = 1$(我們從uniform distribution開始)，也可以從別的情況開始，視你的先驗知識而定
# * 初始情況$N = 1$，那麼正面朝上次數$y$就要等於$\theta$
#
# $$
# \theta ～ beta(\alpha=1,\beta=1)
# $$
#
# $$
# y ～ Bin(N=1, p=\theta)
# $$
#
# * 而我們有4項觀察到的資料需要利用貝氏定理進行推斷來產生後驗分佈(posterior distribution)
#

# 非常直覺的PyMC3，幾乎和數學式一模一樣的程式碼
with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=theta, observed=data)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start=start)

# `find_MAP` optional，有時候能夠為採樣提供一個不錯的初始點，不過大部份時候沒什麼用
# `step`使用Metropolis Hasting來進行採樣，Metropolis Hasting可以進行**離散變量的採樣**

attr_method_list = [attr_method for attr_method in dir(our_first_model) if not attr_method.startswith('_')]
print(attr_method_list)
our_first_model

# ## 診斷採樣過程

# ## 收斂性

# ## 自相關

# ## 總結後驗

# ## 基於後驗的決策

# ## 損失函數




