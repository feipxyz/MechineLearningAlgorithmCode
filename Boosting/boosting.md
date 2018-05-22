# 1. 集成方法概述

## 1.1 集成方法概述

## 1.2 个体学习器

目前来说，同质个体学习器的应用是最广泛的，一般我们常说的集成学习的方法都是指的同质个体学习器。而同质个体学习器使用最多的模型是CART决策树和神经网络。同质个体学习器按照个体学习器之间是否存在依赖关系可以分为两类，第一个是个体学习器之间存在强依赖关系，一系列个体学习器基本都需要串行生成，代表算法是boosting系列算法，第二个是个体学习器之间不存在强依赖关系，一系列个体学习器可以并行生成，代表算法是bagging和随机森林（Random Forest）系列算法。

## 1.3 boosting

## 1.4 bagging
bagging的个体弱学习器的训练集是通过随机采样得到的。通过T次的随机采样，我们就可以得到T个采样集，对于这T个采样集，我们可以分别独立的训练出T个弱学习器，再对这T个弱学习器通过集合策略来得到最终的强学习器。

这里一般采用的是自助采样法（Bootstap sampling）,即对于m个样本的原始训练集，我们每次先随机采集一个样本放入采样集，接着把该样本放回，也就是说下次采样时该样本仍有可能被采集到，这样采集m次，最终可以得到m个样本的采样集，由于是随机采样，这样每次的采样集是和原始训练集不同的，和其他采样集也是不同的，这样得到多个不同的弱学习器。

随机森林是bagging的一个特化进阶版，所谓的特化是因为随机森林的弱学习器都是决策树。所谓的进阶是随机森林在bagging的样本随机采样基础上，又加上了特征的随机选择，其基本思想没有脱离bagging的范畴。



## 1.5 结合策略

### 1.5.1 平均法
对于数值类的回归预测问题，通常使用的结合策略是平均法，也就是说，对于若干个弱学习器的输出进行平均得到最终的预测输出。

最简单的平均是算术平均，也就是说最终预测是

$$
H(x) = \frac{1}{T}\sum\limits_{1}^{T}h_i(x)
$$

如果每个个体学习器有一个权重 $w$，则最终预测是
$$
H(x) = \sum\limits_{i=1}^{T}w_ih_i(x)
$$
其中 $w_i$ 是个体学习器 $h_i$ 的权重，通常有
$$
w_i \geq 0 ,\;\;\; \sum\limits_{i=1}^{T}w_i = 1
$$

# 2. Adaboost
## 2.1 前向分步算法

- 初始化f_0(x) = 0
- 对于m = 1,2,...,M

(1)
  $$
  (\beta_m,\gamma_m) = \arg\min_{\beta,\gamma} \sum_{i=1}^N L(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma))
  $$
(2)
  $$
  f_m(x) = f_{m-1}(x) + \beta_m b(x;\gamma_m) 
  $$

对于回归问题，前向分步算法的损失函数可以选平方损失，即
$$
L(y_i,f(x)) = (y_i - f(x))^2
$$
所以有
$$
L(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma)) = (y_i - f_{m-1}(x_i) - \beta b(x_i;\gamma))^2 \\ = (r_{im} - \beta b(x_i;\gamma))^2
$$
其中 $r_{im}= (y_i - f_{m-1}(x_i))$，这可以理解成是当前模型的残差，为了获取 $\beta_m b(x;\gamma_m)$，也就是令其去拟合当前模型的残差。

而AdaBoost是个分类器，对于分类问题，平方损失就不太适合了。所以引入指数损失，即
$$
L(y,f(x)) = exp(-y f(x))
$$

## 2.2 前向分步算法与adaboost

我们的算法是通过一轮轮的弱学习器学习，利用前一个弱学习器的结果来更新后一个弱学习器的训练集权重。也就是说，第k-1轮的强学习器为
$$
f_{k-1}(x) = \sum\limits_{i=1}^{k-1}\alpha_iG_{i}(x)
$$
而第k轮的强学习器为
$$
f_{k}(x) = \sum\limits_{i=1}^{k}\alpha_iG_{i}(x)
$$
上两式一比较可以得到
$$
f_{k}(x) = f_{k-1}(x) + \alpha_kG_k(x)
$$
可见强学习器的确是通过前向分步学习算法一步步而得到的。

弱分类器系数为：$\alpha_k = \frac{1}{2}log\frac{1-e_k}{e_k}$ 

$G_m(x) \in \lbrace-1,1\rbrace$ 。
则在指数损失的基础上，就需要解决如下问题
$$
(\beta_m,G_m) = \arg\min_{\beta,G} \sum_{i=1}^N exp[-y_i(f_{m-1}(x_i)+\beta G_(x_i))]
$$
令 $w_i^{(m)} = exp(-y_i f_{m-1}(x_i))$，则上述公式可以写成
$$
(\beta_m,G_m) = \arg\min_{\beta,G} \sum_{i=1}^N w_i^{(m)} exp(-\beta y_i G(x_i))
$$
因为 $y_i \in \lbrace-1,1\rbrace$，且 $y_i$ 要么等于 $G(x_i)$，要么不等于。所以将上述公式拆成两部分。暂时省略 $\arg\min$ 之前的部分，exp简写成e，有
$$
e^{-\beta} \sum_{y_i=G(x_i)} w_i^{(m)} + e^{\beta} \sum_{y_i \ne G(x_i)} w_i^{(m)}
$$
在这基础上再添上两项，有
$$
e^{-\beta} \sum_{y_i=G(x_i)} w_i^{(m)} + e^{\beta} \sum_{y_i \ne G(x_i)} w_i^{(m)} + e^{-\beta} \sum_{y_i \ne G(x_i)} w_i^{(m)} - e^{-\beta} \sum_{y_i \ne G(x_i)} w_i^{(m)}
$$
再进一步合并，得到
$$
(e^{\beta} - e^{-\beta}) \sum_{i=1}^N w_i I(y_i \ne G(x_i)) + e^{-\beta} \sum_{i=1}^N w_i^{(m)}     \tag 1
$$
对于迭代的第m步，假设 $\beta$ 为常数，那么公式的右边以及 $(e^{\beta}-e^{-\beta})$都可以看成常数，则要让损失函数取得最小值，只需要让 $\sum_{i=1}^N w_i I(y_i \ne G(x_i))$ 取最小值。因此有
$G_m = \arg\min_G \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))$

那么 $\beta_m$ 怎么求得呢？现在假设 $G_m$ 已知的情况下，回到公式(1)。此时的变量为 $\beta$，要让损失函数取得最小值，先对$\beta$求偏导，得到
$$
\frac {\partial_L} {\partial_{\beta}} = e^{\beta} \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i)) + e^{-\beta} \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i)) - e^{-\beta} \sum_{i=1}^N w_i^{(m)}
$$
再令 $\frac {\partial_L} {\partial_{\beta}} = 0$，得
$$
e^{\beta} \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i)) = [\sum_{i=1}^N w_i^{(m)} - \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))] e^{-\beta}
$$

对两边同求$log$，得到
$$
log \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i)) + log e^{\beta} = log  [\sum_{i=1}^N w_i^{(m)} - \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))] + log e^{-\beta}
$$
又因为 $log e^{-\beta} = -log e^{\beta}$，所以有
$$
log e^{\beta} = \frac {1} {2} log \frac {\sum_{i=1}^N w_i^{(m)} - \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))} {\sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))}
$$
所以解得
$$
\beta_m = \frac {1} {2} log \frac {\sum_{i=1}^N w_i^{(m)} - \sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))} {\sum_{i=1}^N w_i I(y_i \ne G(x_i))}
$$
又因为加权误差率
$$
err_m = \frac {\sum_{i=1}^N w_i^{(m)} I(y_i \ne G(x_i))} {\sum_{i=1}^N w_i^{(m)}}
$$
所以$\beta_m$可以写成
$$
\beta_m = \frac {1} {2} log \frac {1 - err_m} {err_m}
$$
求出了 $G_m(x)$ 与 $\beta_m$ ，就可以写出$f(x)$的更新公式了
$$
f_m(x) = f_{m-1}(x) + \beta_m G_m(x)
$$
根据 $w_i^{(m)} = exp(-y_i f_{m-1}(x_i))$，可以写出$w$的更新公式
$$
w_i^{(m+1)} = exp(-y_i f_m (x_i)) \\ = exp(-y_i (f_{m-1}(x_i)+\beta_m G_m(x_i))) \\ = w_i^{(m)} exp(- \beta_m y_i G_m(x_i))
$$

这与adaboost算法的权值更新只相差规范化因子，因而一致。

到这里也就推导出了当前向分步算法的损失函数选为指数损失的时候，前向分步算法也就是AdaBoost啦。

[参考：从前向分步算法推导出AdaBoost](https://blog.csdn.net/thriving_fcl/article/details/50877957)

## 2.3 adaboost的主要缺点
1）对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性。

# 3. 提升树  
## 3.1 提升树概述  

加法模型、前向分步算法、cart树、可做分类、可做回归。   
提升树被认为是统计学习中性能最好的方法之一。

### 分类问题：  
与adaboost一样  

### 回归问题：  

回归问题采用平方误差损失函数
$$
L(y,f(x))=(y-f(x))^2
$$
按照前向分步算法极小化损失函数，则损失为
$$
\begin{align}L(y,f_{m-1}(x)+T(x;\Theta_m))&=[y-f_{m-1}(x)-T(x;\Theta_m)]^2\\&=[r-T(x;\Theta_m)]^2\end{align}
$$
这里 $r=y-f_{m-1}(x)$ .  
所以回归问题的提升树算法需要计算残差并拟合残差 。

算法：  
$\qquad$ 输入：训练数据集 $T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\},x_i\in\Bbb R^n,y_i\in\Bbb R;$  
$\qquad$ 输出：提升树 $f_M(x)$ .  
$\qquad$ (1) 初始化 $f_0(x)=0$  
$\qquad$ (2) 对 $m=1,2,\cdots,M$  
$\qquad\quad$ (a) 计算残差  
$$
r_{mi}=y_i-f_{m-1}(x_i),\quad i=1,2,\cdots,N
$$
$\qquad\quad$ (b) 拟合残差 $r_{mi}$ 学习一个回归树，得到 $T(x;\Theta_m)$  
$\qquad\quad$ (c) 更新  
$$
f_m(x)=f_{m-1}(x)+T(x;\Theta_m)
$$
$\qquad$ (3) 得到回归问题提升数  
$$
f_M(x)=\sum_{m=1}^MT(x;\Theta_m)
$$

## 3.2 GBDT  
提升树算法利用加法模型与前向分步算法实现学习的优化过程。虽然当损失函数时平方损失和指数损失函数时，每一步的优化很简单，但对于一般损失函数而言，往往每一步的优化并不那么容易。而梯度提升(gradient boosting)算法就是解决这个问题的。

### 3.2.1 GBDT回归   
输入：训练集样本$T=\{(x_,y_1),(x_2,y_2), ...(x_m,y_m)\}$， 最大迭代次数$T$, 损失函数$L$。  
输出：强学习器$f(x)$

1) 初始化弱学习器
$$
f_0(x) = \underbrace{arg\; min}_{c}\sum\limits_{i=1}^{m}L(y_i, c)
$$
2) 对迭代轮数$t=1,2,...T$有：  
$\qquad$ a)对样本$i=1,2，...m$，计算负梯度
$$
r_{ti} = -\bigg[\frac{\partial L(y_i, f(x_i)))}{\partial f(x_i)}\bigg]_{f(x) = f_{t-1}\;\; (x)}
$$
$\qquad$ b)利用$(x_i,r_{ti})\;\; (i=1,2,..m)$, 拟合一颗CART回归树,得到第t颗回归树，其对应的叶子节点区域为$R_{tj}, j =1,2,..., J$。其中$J$为回归树t的叶子节点的个数。

$\qquad$ c) 对叶子区域$j =1,2,..J$,计算最佳拟合值
$$
c_{tj} = \underbrace{arg\; min}_{c}\sum\limits_{x_i \in R_{tj}} L(y_i,f_{t-1}(x_i) +c)
$$
$\qquad$ d) 更新强学习器
$$
f_{t}(x) = f_{t-1}(x) + \sum\limits_{j=1}^{J}c_{tj}I(x \in R_{tj})
$$
$\qquad$ 3) 得到强学习器$f(x)$的表达式
$$
f(x) = f_T(x) =f_0(x) + \sum\limits_{t=1}^{T}\sum\limits_{j=1}^{J}c_{tj}I(x \in R_{tj})
$$

### 3.2.2 GBDT分类  
GBDT的分类算法从思想上和GBDT的回归算法没有区别，但是由于样本输出不是连续的值，而是离散的类别，导致我们无法直接从输出类别去拟合类别输出的误差。

为了解决这个问题，主要有两个方法，一个是用指数损失函数，此时GBDT退化为Adaboost算法。另一种方法是用类似于逻辑回归的对数似然损失函数的方法。也就是说，我们用的是类别的预测概率值和真实概率值的差来拟合损失。下面讨论对数似然损失函数的GBDT分类。而对于对数似然损失函数，又有二元分类和多元分类的区别。

#### 3.2.2.1 二元GBDT分类  