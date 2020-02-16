# Outline
* 機率編程
* 推斷引擎
* PyMC3指南
* 計算方法看拋硬幣問題
* 模型檢查與診斷
  
## 機率編程

貝葉斯統計的概念很簡單
固定的數據，感興趣的參數 -> 探索這些參數可能的取值

我們有先驗$p(\theta)$透過Data轉換為後驗$p(\theta | D)$，換句話說，貝葉斯統計就是一種機器學習的過程 d

然而有一件困擾的事，後驗分佈不一定像是coin toss這個例子一樣具有解析解，很多時候可能是數值解的分佈，也就是說我們要能夠對一個不具解析解的分佈做估計，並瞭解該分佈的行為

## 推斷引擎

即便無法從分析的角度的得到後驗分佈，我們也有辦法估計出來，有以下方法

1. 非馬可夫方法
* 網格計算
* 二次近似
* 變分法

2. 馬可夫方法
* Metropolis-Hastings算法
* Hamiltonian Monte Carlo / No U-Turn Sampler (NUTS)


如今貝葉斯分析主要通過馬可夫鍊蒙特卡羅(Markov Chain Monte Carlo, MCMC)方法進行，同時，變分方法也越來越流行

#### 非馬可夫方法
* 低維度時非常快
* 能提供粗略近似
* 可以作為馬可夫方法的initial condition
##### 網格計算
* 暴力計算
  * 確認參數的合理區間
  * 在以上區間確定一些網格點
  * 對每個點都計算先驗和可能性，視情況做Normolization
  * 效果 : 採樣點越多，近似效果越好，無限多個點就可以得到準確的後驗，但參數一多，就爆了，基本上沒什麼人用

code1

##### 二次近似
* 二次近似 -> 拉普拉斯方法 -> 正態近似
* 利用高斯分佈來近似後驗，通常很有效，原因是因為後驗分佈眾數附近的區域通常符合高斯分佈，事實上很多情況下就是高斯分佈
  * 先找到近似的高斯分佈的mean
  * 然後估計眾數附近的曲率
  * 用曲率來估計標準差
  * Done

##### 變分法
現代貝葉斯統計大多採用MCMC，不過該方法的缺點是慢，不能很好的並行計算
一種簡單的做法是同時算多個馬可夫鍊，然後合併結果，但是對大多數問題而言這不是一個合適的解決方案

對於較大的Dataset或是計算量很大的可能性函數而言 變分法是一個更好的選擇

這個方法能夠快速得到近似，也可作為Initial point

* 沒有介紹大致步驟，最大缺點是針對每個模型都要設計一個特定的算法，因此並非一個通用的推斷引擎，而是模型相關的
* arxiv上提出一個自動差變分推斷(Automatic Differential Variational Inference ADVI)
  * 對參數進行變換，讓他們位於整個實數軸，例如一個參數限定正數，但求log之後則位於無限大到負無限大
  * 用高斯分佈對無界參數近似(之後要逆轉換回來)
  * 採用某種優化使得高斯近似盡可能接近後驗，該過程通過最大化證據下界(Evidence Lower Bound ELBO)實現，如何測量分佈的相似性則是一個關鍵(猜測是cross entropy, KL divergence之類的)
  * ADVI在PyMC3有實現

#### 馬可夫方法

馬可夫方法整個核心思想在於
在估計後驗時並不會所有點都算一遍(暴力解)，而會考慮該點的機率值，如果區間A的機率是區間B的兩倍，那就會從區間A採樣更多進行計算

MCMC(Markov Chain Monte Carlo)拆解成兩個部分說明

##### 什麼是馬可夫鍊?
[資料 1, Wiki](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)
[資料2, 不知名 Intro](http://episte.math.ntu.edu.tw/articles/mm/mm_09_3_08/page2.html)
[可以參照Data Science Note中的Markov Chain Section](https://github.com/YLTsai0609/DataScience_Note/blob/master/Markov_chain_MCMC.md)
* 一種對於序列問題的描述
當前狀態只與上一部的狀態**有相關**，就稱為馬可夫狀態(基本上記憶性只有一格的意思)
* 與獨立事件不同，講述的是相關，而相關可以跟好幾部以前相關，Markov chain是最簡單且有好數學性質的一種，只和上一個狀態有關

> 當前狀態 --> 狀態轉移 --> 下一狀態

* 從此數據工具可以衍伸個個變種來描述線實況，例如
  * m階馬可夫鏈 : 和前m個狀態有關係

* 隨機漫步即為一個例子

所以馬可夫鍊怎麼樣應用在尋找後驗上呢?

假設我們能夠找到一個馬可夫鍊，其狀態轉移的機率正比於我們想要採樣的分佈，那麼貝葉斯公式就在Markov chain之中了

$$
p(\theta | D) \propto p(D | \theta) p(\theta)
$$
上式就是在描述一個馬可夫關係!

但是要怎麼樣在不知道後驗分佈的情況下找到這樣的狀態鍊呢?
馬可夫鏈中有一個概念叫做Detailed Balance Condition(細平衡條件)
直觀的說，我們需要採用一種可逆的方式移動，從狀態i轉移到狀態j的機率必須和狀態j轉移到狀態i的機率相等(?)

總結的說，如果我們能夠找到滿足Detailed Balance Conditon的馬可夫鍊，就可以保證從採樣中得到的分佈來自於我們的後驗分佈，而保證上述條件目前最流行的算法是Metropolis-Hasting算法

#### Monte Carlo
Monte Carlo是一系列應用非常廣泛的算法，**其思想是通過隨機採樣或是計算麼你給定過程**，Monte Carlo是位於摩洛哥公園一個非常有名的城市，開發者Stanislaw Ulam。Stan正是基於這一核心思想 - **儘管很多問題都難以求解甚至無法精確用公式表達，但我們可以通過採樣或者模擬來有效地研究**
* 使用Monte Carlo方法計算數值的例子中，一個經典例子是估計$\pi$
* 在邊長為$2R$的帳方形內隨機撒$N$個點
* 在正方形內化一個半徑為$R$的圓，計算在圓圈內點的個數
* 得出$\hat{\pi}$的估計值$\frac{4\times inside}{N}$
* inside的counts $\sqrt{(x^2 + y^2)}\leq R$
* 因為正方形的面積是$4R^2$，元的面積是$\pi R^2$所以兩者的比例是$\frac{\pi}{4}$，因此我們可以估計$\hat{\pi}$

#### Metroplis Hasting

#### Hamiton Monte Carlo

#### 其他MCMC
* 總結來說，MCMC目前有許多研究正走向平行化的計算，畢竟MCMC最為人詬病的就是慢!

#### PyMC3
* PyMC3是一個用於機率編程的Python library，PyMC3提供了一套非常簡潔直觀的語法，非常接近統計學中描述機率模型的語法，可讀性很高，核心部分基於Numpy和Theano
* 用PyMC3來解拋硬幣問題(單參數估計)(見ch2.py)
* [使用MCMC時的收斂性以及hint可以參考](https://wangcc.me/LSHTMlearningnote/MCMC-methods.html#%E4%BD%BF%E7%94%A8-mcmc-%E6%99%82%E9%9C%80%E8%A6%81%E8%80%83%E6%85%AE%E7%9A%84%E4%B8%80%E4%BA%9B%E5%95%8F%E9%A1%8C)