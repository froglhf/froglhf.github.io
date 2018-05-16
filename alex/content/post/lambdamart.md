+++
banner = "/banners/ocean_road.jpg"
categories = ["machine learning"]
date = "2018-05-14T02:10:51+02:00"
description = ""
images = []
menu = ""
tags = ["machine learning","ranking"]
title = "LambdaMART简介"
+++

**LambdaMART**是一种state-of-art的Learning to rank算法，由微软在2010年提出[[1]](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)。在工业界，它也被大量运用在各类ranking场景中。LambdaMART可以看做GDBT版本的**LambdaRank**，而后者又是基于**RankNet**发展而来的。RankNet最重要的贡献是提出了一种pairwise的用于排序的概率损失函数，而LambdaRank又在损失函数中巧妙的引入了NDCG等ranking metrics来优化排序效果。LambdaMART则是集大成者，它结合了上述两篇文章中提出的Lambda函数以及GDBT这一学习算法，在各种排序问题中均能取得不错的效果。下面是一些开源实现：

1. [Ranklib](https://sourceforge.net/p/lemur/wiki/RankLib/)
2. [Xgboost](https://github.com/tqchen/xgboost)

`封面： 大洋路沿途`

<!--more-->

## RankNet

RankNet是2005年微软提出的一种pairwise的Learning to rank算法，它从概率的角度来解决排序问题。RankNet提出了一种pairwise的概率损失函数，并可以应用于任意对参数可导的学习算法。在论文中，RankNet基于神经网络实现，除此之外，GDBT等模型也可以应用该损失函数。
RankNet是一个pairwise的算法，它首先将训练数据中同一Query下的doc两两组成pair，用${U_i，U_j}$表示。模型的学习目标是得到一个打分函数$f(x)$，它的输入是某个doc的特征向量$x$，输出是一个实数，值越高代表该doc的排序位置应该越靠前。也就是说，当$f(x_i)>f(x_j)$时，$U_i$的排序位置应该在$U_j$之前，用$U_i \rhd U_j$表示。基于此，我们定义$U_i$比$U_j$排序位置更靠前的概率如下，其中，$s=f(x)$.

$$P(U_i \rhd U_j)=P(s_i>s_j)= \frac{1}{1+e^{\sigma(s_i-s_j)}}$$

我们的目标概率（理想情况，预测概率应该尽可能拟合的概率）如下：

$$\bar{P_{ij}}=\bar{P}(U_i \rhd U_j)=\begin{cases}
1, & U_i \rhd U_j \\\\\\
0, & U_i \lhd U_j \\\\\\
0.5, & equals
\end{cases}$$

为了方便计算，我们令：

$$ \bar{P\_{ij}}=\frac{1}{2}(S\_{ij}+1), 其中S\_{ij}={1, -1, 0}$$

这样，根据$U\_i$和$U\_j$的标注得分，就可以计算$\bar{P\_{ij}}$。

有了目标概率和模型预测概率，使用交叉熵损失函数（cross entropy loss function）作为概率损失函数，它衡量了预测概率和目标概率在概率分布上的拟合程度：

$$C=-\bar{P\_{ij}}logP\_{ij}-(1-\bar{P\_{ij}})\log (1-P\_{ij})=\frac{1}{2}\sigma(s\_i-s\_j)(1-S\_{ij})+\log (1+e^{-\sigma(s\_i-s\_j)})$$

计算C关于模型参数$w_k$的偏导，并应用gradient descent求解：

$$w_k \rightarrow w_k-\frac{\partial C}{\partial w_k}$$

总的来说，RankNet从概率角度定义了排序问题的loss function，并通过梯度下降法求解。所以RankNet依赖的模型必须是平滑的，保证梯度是可以计算的。在paper中，作者选择一个两层的神经网络作为排序模型。除此之外，选择GBDT也可以取得不错的效果。

>**交叉熵**：
设随机变量$X$服从的概率分布为$p(x)$，往往$p(x)$是未知的，我们通过统计方法得到$X$的近似分布$q(x)$，则随机变量$X$的交叉熵为：
$$H=-p(x)\log q(x)-(1-p(x))\log (1-q(x))$$
它衡量了q(x)和p(x)的拟合程度,交叉熵越大则两个分布差异越大

## Mini-Batch

在上述的学习过程中，每一对样本${U_i，U_j}$都会更新一次参数$w$，如果采用BP神经网络模型，每一次更新都需要先前向预测，再误差后向反馈，训练过程非常慢。因此，有了下面的加速算法:

对于给定的样本对${U_i，U_j}$，我们有如下推导：

$$\frac{\partial C}{\partial w_k}=\frac{\partial C}{\partial s_i}\frac{\partial s_i}{\partial w_k}+\frac{\partial C}{\partial s_j}\frac{\partial s_j}{\partial w_k}=\lambda\_{ij}\left (\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k} \right )$$

第二步推导是根据：
$$\frac{\partial C}{\partial s_i}=-\frac{\partial C}{\partial s_j}=\lambda\_{ij}$$

我们用$I$表示在给定query下所有参与训练的文档对{i,j}的集合（显然$U_i$和$U_j$应该有不同的标注）。为了不包含重复pair，我们假设$U_i \rhd U_j$。这样，对某个参数$w_k$累计的更新可以表示为：
$$\delta w_k=-\eta \sum\_{{i,j} \in I}\lambda\_{ij}\left (\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k} \right )=-\eta \sum_i\lambda\_{i}\frac{\partial s_i}{\partial w_k}$$

$$\lambda\_{i}=\sum\_{\\{i,j\\}\in I}\lambda\_{ij}-\sum\_{\\{j,i\\}\in I}\lambda\_{ij}$$

这样，我们就由每计算一个样本对${U_i，U_j}$更新一次w，改为了每次计算$U_i$所能组成的所有样本对，再更新一次w。加速算法可以看成是一种mini-batch的梯度下降算法。

$\lambda_i$具有明确的物理意义，它的正负表示$U_i$在下次更新需要移动的方向，大小表示$U_i$在下次更新需要移动的强度。

## LambdaRank

在RankNet中，我们使用了交叉熵概率损失函数作为优化目标。但ranking任务通常选择NDCG、ERR作为评价指标，这两者间存在一定的mismatch。比如。另一方面，NDCG、ERR是非平滑、不连续的，无法求导，不能直接运用梯度下降法求解，将其直接作为优化目标是比较困难的。因此，LambdaRank选择了直接定义cost function的梯度来解决上述问题。
LambdaRank是一个经验算法，它直接定义的了损失函数的梯度lambda函数。Lambda函数由两部分相乘得到：(1)RankNet中交叉熵概率损失函数的梯度；(2)交换${U_i，U_j}$位置后评价指标Z的差值的绝对值。具体如下：

$$\lambda\_{ij}=\frac{-\sigma|\Delta Z\_{ij}|}{1+e^{\sigma(s_i-s_j)}}$$
$Z$可以是NDCG、ERR、MRR、MAP等ranking评价指标

损失函数的梯度代表了文档下一次迭代优化的方向和强度，由于引入了Ranking评价指标，Lambda函数更关注位置靠前的高质量文档的排序位置的提升。有效的避免了在迭代中下调高质量文档这种情况的发生。

LambdaRank相比RankNet的优势在于引入了ranking评价指标，更符合实际需要，有人也把LambdaRank/LambdaMART看作listwise的方法。

## LambdaMART

LambdaRank中重新定义了损失函数的梯度，而这个Lambda梯度可以应用于任何可以使用梯度下降法求解的模型。自然，我们想到了将Lambda梯度和MART结合，这就是LambdaMART。

对于LambdaMART有如下优点：

1. 直接对排序问题进行优化，而非将其转换为分类或回归问题；
2. GBDT具有比较好的泛化能力，经受了实践的考验；
3. 引入ranking评价指标作为优化目标
3. 易于Continue training；
4. GBDT具有特征选择的能力，可以反映不同特征的重要程度；
5. 有较好的分布式实现方案