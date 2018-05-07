---
title: "Test"
date: 2018-05-03T19:43:12+08:00
banner: "/banners/hutong.png"
categories: ["deep learning"]
description: ""
images: []
menu: ""
tags: ["deep learning"]
title: "Deep Learning Chapter 1 - Introduction"
---

正式拔草《Deep Learning》，相关资源 http://www.deeplearningbook.org/

`封面： 北京胡同`

<!--more-->

---

## What's Deep Learning
一个核心观点是，深度学习是是一种学习数据表示（data representtation）的机器学习技术。

在运用传统机器学习技术（如LR、SVM）解决实际问题时，我们并不将原始数据输入机器学习算法，而是将数据表示成特定的形式，这一步通常被称作特征抽取。算法的性能很大程度上依赖于特征的质量，但手工构造特征具有以下局限性：

 * 构造好的特征需要研究人员丰富的领域知识和深厚的特征工程经验。
 * 特征构造是在充分理解数据的基础上进行的，如果训练数据发生较大的变化，可能需要重新构造特征集合。
 * 对于一部分概念和抽象，很难手工选择合适的特征来表示。
	
一个解决方法是让机器自己去学习数据的表示，这种方法被称为表示学习。表示学习的一个例子是autoencoder。autoencoder包含一个encoder函数和一个decoder函数，encoder函数可以将输入数据转换成某种新的表示，并且这个表示具有一些有用的特性。而decoder函数将表示还原为原始输入，还原的数据与原始数据尽可能的保持一致。

但是，传统的表示学习方法很难学习到隐含在数据背后的抽象特征。而深度学习技术则通过简单的表示来学习复杂表示，从而很好的解决这个问题。（深度学习技术应用于图像识别的例子）


## Historical Trends in Deep Learning