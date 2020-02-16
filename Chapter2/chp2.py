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

# save = input("Save the figure? y/n")
# if save == 'y':
#     plt.savefig(FILE_PATH_brute_force, dpi=300, figsize=(5.5, 5.5));
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

# save = input("Save the figure? y/n")
# if save == 'y':
#     plt.savefig(FILE_PATH_monte_carlo, dpi=300, figsize=(5.5, 5.5))


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

# save = input("Save the figure? y/n")
# if save == 'y':
#     plt.savefig(FILE_PATH_metropolis_hasting, dpi=300, figsize=(5.5, 5.5))
# -
# # PyMC3 for Coin toss


# 二項分佈，real value 取0.35
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
#     y = pm.Binomial('y', p=theta, observed=data)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start=start)

# `find_MAP` optional，有時候能夠為採樣提供一個不錯的初始點，不過大部份時候沒什麼用
# `step`使用Metropolis Hasting來進行採樣，Metropolis Hasting可以進行**離散變量的採樣**

attr_method_list = [attr_method for attr_method in dir(our_first_model) if not attr_method.startswith('_')]
print(attr_method_list)
our_first_model

# ## 診斷採樣過程
# ## sampling step是針對後驗進行採樣，在解析解種對應是什麼?
# ### Review analytic solution
#     * 在解析解的過程中，針對每一組資料點我們可以按照可能性以及先驗計算出後驗，持續迭代後驗，該行為可以被一次計算，因為有了試驗次數$N$以及試驗成功次數$y$，假定可能性函數為二項分佈的情況下，我們就能夠計算出可能性分佈
# $$
# p(\theta | y) = \frac{N!}{y!(N-y)!}\theta^{y}(1-\theta)^{N-y}
# $$
#     <img src = '../images/bay_chp1_3.png'></img>
#     * 然後一次性的計算出後驗分佈
# $$
# p(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}\theta^{\alpha - 1}(1-\theta)^{\beta - 1}
# $$
#
# $$
# p(\theta | y) \propto \theta^{\alpha -1 + y}(1-\theta)^{\beta -1 +N -y}
# $$
#     * 只要把數字帶進去我們就找到後驗分佈了
# ### Ans
# * 既然是可以被一次性計算的，抽樣次數越多其實就是越貼近後驗分佈，所以抽樣次數少，得到的後驗分佈更rough，抽樣次數多，得到的後驗分佈更detail，既然如此，我們能否知道**MCMC抽樣何時收斂? 抽多少sample即可?**

# ## 收斂性
# * [參考](https://wangcc.me/LSHTMlearningnote/MCMC-methods.html#%E4%BD%BF%E7%94%A8-mcmc-%E6%99%82%E9%9C%80%E8%A6%81%E8%80%83%E6%85%AE%E7%9A%84%E4%B8%80%E4%BA%9B%E5%95%8F%E9%A1%8C)
# * initial value - 這些起始值是用來輔助 MCMC 採樣的，起始值並不是先验概率(initial values are not priors)。
# * 收斂時間 - 需要多少樣本才能讓樣本$\theta^{t}$接近後驗分佈$p(\theta|x)$
# * 收斂效率 - 採集的樣本$\theta^{t}$是否在估計$p(\theta|x)$能夠有效的估計
#
# ### 收斂時間
# 沒有人能夠準確地說出MCMC的採樣100%地達到收斂，然而關於判斷，確實有一些準則
# * 視覺檢查
# * 剔除不穩定的樣本(burn-in iterations)
# * 指標計算 - 透過MCMC樣本的鏈內方差(within chain)和鏈間方差(between chain)來進行收斂診斷
# 一般來說三者都會用，來確定說目前的MCMC很接近收斂了
#
# #### 具體來說
# * 把MCMC事後概率分布採樣過程的整個歷史(history)痕跡(trace)全部繪制出來-不同起始值的同一個未知參數的MCMC鏈是否都給出了相對穩定的歷史痕跡？他們是否有合理的相互重疊(overlapping)？
# * $theta^{i}$分佈應該看起來要類似高斯分佈，因為有中央極限定理，但是如果$N$很小，那就未必
# * $theta^{i}$隨著iteration應該要看起來像white noise，表示震盪逐漸趨於隨機(意味著收斂)
# * 檢查自我相關程度(autocorrelation)-過高的自相關暗示收斂過程較慢。(high autocorrelation is a symtom of slow convergence)。
# * 看Gelman-Rubin收斂統計量-它通過比較MCMC鏈內方差(within variability)和鏈間方差(between variability)來評估MCMC鏈是否達到收斂。
#
# #### 參數太多時
# * 隨機選取幾個來分析其結果是否收斂

attrs = [attr for attr in dir(chain) if not attr.startswith('_')]
print(attrs, chain.chains, chain.get_values,
      chain.point, chain.points,
      chain.add_values,
      chain.get_sampler_stats,
      sep='\n\n')
# check current theta
# for i in range(0,1000, 50):
#     print('-'*60)
#     print(i)
#     print(chain.point(i), chain.point(i+1))

# * 每經過一次採樣，會得到'01100011', '1010100010010'的序列，每次都可以重新計算一次$\theta^{i}$，所以經過1000次，我們就可以有1000個$\theta^{i}, i \in \{0, N \}$
# * 下左圖是$\theta^{i}$的KDE plot of histogram
# * 下右圖是$\theta^{i}$隨著iteration的震盪值

# +
print(start, step, trace, sep='\n\n')
burnin = 0  # no burnin
chain = trace[burnin:]
ax = pm.traceplot(chain, lines={'theta':theta_real}); # return 兩個ax, 分別是左圖跟右圖
for sub_ax in ax[0]:
    sub_ax.set_title('theta no burnin')



print(start, step, trace, sep='\n\n')
burnin = 250  # no burnin
chain = trace[burnin:]
pm.traceplot(chain, lines={'theta':theta_real});
for sub_ax in ax[0]:
    sub_ax.set_title('theta burnin = 250')

# -

# * 尚未收斂的$\theta$可能會在$\theta^{i}$ 和 iteration的子圖中展示出趨勢，這不是我們要的，我們希望看到white noise表示收斂了
# <img src='../images/bay_chp2_1.png'></img>

# # gelman rubin檢定可以說明什麼?
# * 對於多條不同的MCMC trace，由於不同的trace是從不同的初始點開始，而且過程彼此是獨立的，所以可以確認多個trace中在$\theta$ vs iteration的子圖中是否互相重疊，是否皆為white noise，而kde plot是否有類似的分佈，對於一個趨近於穩定的分佈
# * trace1和trace2的表現應該要差不多，我們從within-chain variance以及between-chain variance來看，即Gelman-Rubin檢驗，理想狀態下，檢驗值應該要是1，我們收到的值如果低於1.1，那麼就可以認為是收斂的了，更高的值則意味著沒有收斂
#
# ## step of gelman rubin
# 1. $m$th chain, $N_{m}$ interations : $\theta^{m}_{1}, \theta^{m}_{2}, \theta^{m}_{3}, ...\theta^{m}_{N_{m}}$
# 2. 對於每個parameter chain $\theta$, 後驗平均(posterior mean $\hat{\theta_{m}} = \frac{1}{N_{m}} \sum_{i}^{N_{m}} \theta^{m}_{i}$)
# 3. 對於每個parameter，chain $\theta$ 後驗方差(intra-chain variance)
# $\sigma^{2}_{m} = \frac{1}{N_{m}-1} \sum_{i}^{N_{m}} (\theta^{m}_{i} - \hat{\theta^{m}})^{2}$
# 4. 計算parameter mean ay chains level $\hat{\theta} = \frac{1}{M} \sum_{m}^{M}\hat{\theta_{m}}$
# 5. Compute how indiv. means scatter around the joint mean
# (計算一個數值$B$，類似每個chain的平均$\hat{\theta_{m}}$距離對於chain的平均$\hat{\theta}$的變異數，再進行伸縮)
# $B = \frac{N}{M-1} \sum_{m=1}^{M} (\hat{\theta_{m}} - \hat{\theta})^{2}$
# 6. 計算variance mean at chains level
# $W = \frac{1}{M} \sum_{m=1}^{M} \sigma^{2}_{m}$
# 7. 計算$\hat{V} = \frac{N-1}{N}W + \frac{M+1}{MN}B$
# 8. test whether $R=\sqrt{\hat{V}/W}～1$
#
# * TBD....
# [google key words](https://www.google.com/search?safe=strict&sxsrf=ACYBGNRe2B7cbde-AdozJFSWvBTQ4L5LpQ%3A1581845967510&ei=zw1JXuvZHo2JmAXVypDgCQ&q=gelman+rubin+explained+medium&oq=gelman+rubin+explained+medium&gs_l=psy-ab.3...161688.162555..162701...0.0..0.158.1067.0j9......0....1..gws-wiz.......33i160j33i22i29i30j33i21.z4DxADGTU2I&ved=0ahUKEwir_e2349XnAhWNBKYKHVUlBJwQ4dUDCAs&uact=5)
#
# [Gelman–Rubin convergence diagnostic using multiple chains
# ](https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/)
#
# [Convergence tests for MCMC](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/astrophysics/public/icic/data-analysis-workshop/2018/Convergence-Tests.pdf)

pm.gelman_rubin(chain)


# ## 關於模型效率

# # 兩個$\theta^{i}$的自相關性，如何解釋MCMC會造成自相關性?

pm.autocorrplot(chain)

pm.forestplot(chain, varnames=['theta']);

pm.stats.summary(chain)

# ## 總結後驗

pm.effective_n(chain)

pm.plot_posterior(chain)

# ## 基於後驗的決策

pm.plot_posterior(chain, rope=[0.45, 0.55])

pm.plot_posterior(chain, ref_val=0.5)

# ## 損失函數




