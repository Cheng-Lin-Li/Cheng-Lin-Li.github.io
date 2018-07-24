---
layout: post
title: Hello Data !!
tags: ai machine-learning nlp concept
---

<!-- more -->

---
Table of content:

{: class="table-of-content"}
* TOC
{:toc}

---

## What is the major focus of this blog?

This is my learning notebook which includes AI, Machine Learning, Big Data techniques, Knowledge Graph, Information Visualization and Natural Language Processing.

I am a lifelong learner and passionate to contribute my knowledge to impact the world. After twenty years dedicate working on IT Application development departments for Intranet, B2C, B2B eCommerce portal and trading Websites, I observed the A.I. and DATA era is coming. It is time to let DATA tell its story by A.I. / machine learning algorithms and that's the reason why I resigned from J.P. Morgan Asset Management (Taiwan) in 2016 and went back to school to be a graduate student in Viterbi school of engineering at University of Southern California. My research area is data informatics and I would like to share what I learn with everyone.

I will leverage my spare time to enrich this notebook style blog from time to time. Your comments are appreciated.

---
## Reference material:

### Artificial intelligence (AI)

Textbook:
1. [Stuart Russell,‎ Peter Norvig, Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)

---

### Machine Learning (ML)
Textbook:
1. [Introduction to Machine Learning-3rd, Ethem Alpaydin](https://mitpress.mit.edu/books/introduction-machine-learning-0)
2. [The Elements of Statistical Learning: Data Mining, Inference, and Prediction (Second Edition), by Trevor Hastie, Robert Tibshirani and Jerome Friedman](https://web.stanford.edu/~hastie/pub.htm)
3. [Pattern Recognition And Machine Learning, Bishop](https://www.microsoft.com/en-us/research/people/cmbishop/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fcmbishop%2Fprml%2F)
4. [Deep Learning, Ian Goodfellow and Yoshua Bengio and Aaron Courville](http://www.deeplearningbook.org/)

Articles & Papers:
1. [Deep Learning: An Introduction for Applied Mathematicians](https://arxiv.org/abs/1801.05894v1)
2. [The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches](https://arxiv.org/abs/1803.01164)

Training Material and Courses:
1. [CS229: Machine Learning](http://cs229.stanford.edu/syllabus.html)
2. [Representation learning in Montreal Institute for Learning Algorithms @Universite' de Montre'al](https://ift6135h18.wordpress.com/) (Deep Learning)
3. [Reinforcement Learning: An Introduction by Prof. Richard S. Sutton & Andrew G. Barto @University of Alberta](https://drive.google.com/drive/folders/0B3w765rOKuKANmxNbXdwaE1YU1k), [OR try this alternative link.](http://www.incompleteideas.net/book/the-book-2nd.html)
4. [Carnegie Mellon University - 10715 Advanced Introduction to Machine Learning: lectures](https://sites.google.com/site/10715advancedmlintro2017f/lectures)
5. [Deeplearning.ai, Andrew Ng, Introductory deep learning course.](https://www.deeplearning.ai/)

---
### Natural Language Processing(NLP)
Articles & Papers:
1. [Demystifying, word2vec](https://www.deeplearningweekly.com/blog/demystifying-word2vec)
2. [Brill (1992): A Simple Rule-Based Part of Speech Tagger](http://www.aclweb.org/anthology/A/A92/A92-1021.pdf)
3. [Ratnaparkhi (1996): A Maximum Entropy Model for Part-Of-Speech Tagging](http://www.aclweb.org/anthology/W/W96/W96-0213.pdf)
4. [Lafferty, McCallum and Pereira (2001): Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?referer=http://ron.artstein.org/csci544-2018/index.html&httpsredir=1&article=1162&context=cis_papers)
5. [Young (1996): A review of large-vocabulary continuous-speech recognition. IEEE Signal Processing Magazine 13(5): 45–57.](http://ieeexplore.ieee.org/document/536824/)
6. [Sutskever, Vinyals and Le (2014): Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
7. [Neubig (2017): Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf)
8. [Mikolov, Yih and Zweig (2013): Linguistic Regularities in Continuous Space Word Representations](https://aclweb.org/anthology/N/N13/N13-1090.pdf)
9. [Levy, Goldberg and Dagan (2015): Improving Distributional Similarity with Lessons Learned from Word Embeddings.](https://aclweb.org/anthology/Q/Q15/Q15-1016.pdf)


Training Material and Courses:
1. [Natural Language Processing (Fall 2017) by Prof. Jason Eisner @Johns Hopkins University](http://www.cs.jhu.edu/~jason/465/)
2. [Natural Language Processing with Deep Learning (Winter 2017) by Chris Manning & Richard Socher @Standford University: Material website](http://cs224d.stanford.edu/) [,and video link](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)

---

### Statistics:
[Statistics and R by Nathaniel E. Helwig@University of Minnesota](http://users.stat.umn.edu/~helwig/teaching.html)

---

## General topics
> I leave some Technology notes in this section. I may write articles for each of them in the future.

#### Parametric vs. Nonparametric Methods.
##### reference:
1. [Stuart Russell,‎ Peter Norvig, Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
* Parametric Methods: 

A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a parametric model. No matter how much data you throw at a parametric model, it won’t change its mind about how many parameters it needs.

  Models do not growth with data.
  
  Model examples:

  >1-1. Linear regression
  >
  >1-2. Logistic regression
  >
  >1-4. Perceptron
  >
  >1-5. Naive Bayes
  >
  >1-6. ...etc.


* Nonparametric Methods: Don't summarize data into parameters.

Nonparametric methods are good when you have a lot of data and no prior knowledge, and when you don’t want to worry too much about choosing just the right features.

  Models growth with data.
  
  Model examples:
  >1-1. k-nearest neighbors
  >
  >1-2. Support Vector Machine
  >
  >1-3. Decision Tree: (CART and C4.5)

---
#### Generative & Discriminative models:
##### Reference:
1. [https://en.wikipedia.org/wiki/Generative_model](https://en.wikipedia.org/wiki/Generative_model)


* Generative model, also called joint distribution models.

  Generative learning algorithms assume there is a model to GENERATE the observable variable by hidden(or target) variable and the hidden variables is a distribution rather than a fix value.
 
  Given an observable variable X and a target variable Y, a generative model is a statistical model of the joint probability distribution on X × Y, P ( X , Y )

  1. Gaussian mixture model and other types of mixture model
  2. Hidden Markov model
  3. Probabilistic context-free grammar
  4. Naive Bayes
  5. Averaged one-dependence estimators
  6. Latent Dirichlet allocation
  7. Restricted Boltzmann machine
  8. Generative adversarial networks

* Discriminative model, also called conditional models.

  A discriminative model is a model of the conditional probability of the target Y, given an observation x, symbolically, P ( Y \| X = x ) and, 
  
  Classifiers computed without using a probability model are also referred to loosely as "discriminative".

  Algorithms that try to learn P( Y \| X ) directly (such as logistic regression) by given X, or algorithms that try to learn mappings directly from the space of inputs X to the labels {0,1}, (such as the perceptron algorithm) are called discriminative learning algorithms. 

  1. Logistic regression, a type of generalized linear regression used for predicting binary or categorical outputs (also known as maximum entropy classifiers)
  2. Support vector machines
  3. Boosting (meta-algorithm)
  4. Conditional random fields
  5. Linear regression
  6. Neural networks
  7. Random forests

---
#### Look-Ahead Bias
##### Reference:
1. [https://www.investopedia.com/terms/l/lookaheadbias.asp](https://www.investopedia.com/terms/l/lookaheadbias.asp)

Look-ahead bias occurs by using information or data in a study or simulation that would not have been known or available during the period being analyzed. This will usually lead to inaccurate results in the study or simulation. Look-ahead bias can be used to sway simulation results closer into line with the desired outcome of the test.

To avoid look-ahead bias, if an investor is backtesting the performance of a trading strategy, it is vital that he or she only [uses information that would have been available at the time of the trade]. For example, if a trade is simulated based on [information that was not available] at the time of the trade - such as a quarterly earnings number that was released three months later - it will diminish the accuracy of the trading strategy's true performance and potentially bias the results in favor of the desired outcome. 
Look-ahead bias is one of many biases that must be accounted for when running simulations. Other common biases are :

a. [sample selection bias]: Non-random sample of a population, 

b. [time period bias]: Early termination of a trial at a time when its results support a desired conclusion.

c. [survivorship/survival bias]: It is the logical error of concentrating on the people or things that made it past some selection process and overlooking those that did not, typically because of their lack of visibility.


All of these biases have the potential to sway simulation results closer into line with the desired outcome of the simulation, as the input parameters of the simulation can be selected in such a way as to favor the desired outcome.

---
#### Ensemble Learning to Improve Machine Learning Results
Reference:

  1. Vadim Smolyakov, [Ensemble Learning to Improve Machine Learning Results.](https://blog.statsbot.co/ensemble-learning-d1dcd548e936)
  
  2. [Bagging, boosting and stacking in machine learning](https://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning)

Ensemble methods are meta-algorithms which combine several machine learning techniques into one model to increase the performance:

  1. bagging (decrease variance): bootstrap aggregation. Parallel ensemble: each model is built independently
    a. Reduce the variance of an estimate is to average together multiple estimates.
    b. Bagging uses bootstrap sampling (combinations with repetitions) to obtain the data subsets for training the base learners. For aggregating the outputs of base learners, bagging uses voting for classification and averaging for regression.

  2. boosting (decrease bias): Sequential ensemble: try to add new models that do well where previous models lack.
    a. Boosting refers to a family of algorithms that are able to convert weak learners to strong learners. The main principle of boosting is to fit a sequence of weak learners− models that are only slightly better than random guessing, such as small decision trees− to weighted versions of the data. More weight is given to examples that were misclassified by earlier rounds.
    b. Two-step approach, where first uses subsets of the original data to produce a series of averagely performing models and then "boosts" their performance by combining them together using a particular cost function (majority vote for classification or a weighted sum for regression). Unlike bagging, in the classical boosting the subset creation is not random and depends upon the performance of the previous models: every new subsets contains the elements that were (likely to be) misclassified by previous models.

  3. stacking (improve predictions): Sequential ensemble: stacking is an ensemble learning technique that combines multiple classification or regression models via a meta-classifier or a meta-regressor. 
    a. The base level models are trained based on a complete training set, then the meta-model is trained on the outputs of the base level model as features.

---
####  Glorot initialization/ Xavier initialization
##### References:
1. [http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
2. [https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/](https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/)

Glorot initialization: it helps signals reach deep into the network.
	
  a. If the weights in a network start too small, then the signal shrinks as it passes through each layer until it’s too tiny to be useful.
	
  b. If the weights in a network start too large, then the signal grows as it passes through each layer until it’s too massive to be useful.

Formular: $$Var(W) = \frac{1}{n_{in}}$$

where W is the initialization distribution for the neuron in question, and n_in is the number of neurons feeding into it. The distribution used is typically Gaussian or uniform.

It’s worth mentioning that Glorot & Bengio’s paper originally recommended using: 
	$$Var(W) = \frac{2}{(n_{in}+n_{out})}$$
	where $$n_{out}$$ is the number of neurons the result is fed to.

---
####  He initialization: For the more recent rectifying nonlinearities (ReLu)
##### References:
1. [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)

Formular: $$Var(W) = \frac{2}{n_{in}}$$

Which makes sense: a rectifying linear unit is zero for half of its input, so you need to double the size of weight variance to keep the signal’s variance constant.

---
####  GloVe: Global Vectors for Word Representation
##### References:
1. [Jeffrey Pennington, Richard Socher, Christopher D. Manning, GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

	GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

___
####  $$F_{\beta}$$ score: An easy to combine precision and recall measures
>
>$$F_{\beta} = \frac{(1+\beta^2)(precision*recall)}{(\beta^2*precision+recall)}$$
>
>$$F1 = \frac{2*(precision*recall)}{(precision+recall)}$$

___
#### Symmetric Mean Absolute Percent Error (SMAPE)
##### References:
1. [http://www.vanguardsw.com/business-forecasting-101/symmetric-mean-absolute-percent-error-smape/](http://www.vanguardsw.com/business-forecasting-101/symmetric-mean-absolute-percent-error-smape/)

	An alternative to Mean Absolute Percent Error (MAPE) when there are zero or near-zero demand for items. SMAPE self-limits to an error rate of 200%, reducing the influence of these low volume items. Low volume items are problematic because they could otherwise have infinitely high error rates that skew the overall error rate.
	SMAPE is the forecast minus actual divided by the sum of forecasts and actual as expressed in formula:
  

> $$SMAPE = \frac{2}{N} * \sum_{k=1}^N\frac{\vert F_k-A_k\vert}{(F_k + A_k)}$$
>
>k = each time period.

####  Mean Absolute Percent Error (MAPE)
##### References:
1. [http://www.vanguardsw.com/business-forecasting-101/mean-absolute-percent-error/](http://www.vanguardsw.com/business-forecasting-101/mean-absolute-percent-error/)

Mean Absolute Percent Error (MAPE) is the most common measure of forecast error. MAPE functions best when there are no extremes to the data (including zeros).

With zeros or near-zeros, MAPE can give a distorted picture of error. The error on a near-zero item can be infinitely high, causing a distortion to the overall error rate when it is averaged in. For forecasts of items that are near or at zero volume, Symmetric Mean Absolute Percent Error (SMAPE) is a better measure.
	MAPE is the average absolute percent error for each time period or forecast minus actuals divided by actuals:

> $$MAPE = \frac{1}{N} * \sum_{k=1}^N\frac{\vert F_k-A_k\vert}{A_k}$$
>
>k = each time period.

---

####  MLE vs MAP: the connection between Maximum Likelihood and Maximum A Posteriori Estimation
##### References:
1. [http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/MLE_MAP_Part1.pdf](http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/MLE_MAP_Part1.pdf)
2. [https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/](https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/)
	
  Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP), are both a method for estimating some variable in the setting of probability distributions or graphical models. They are similar, as they compute a single estimate, instead of a full distribution.
	Maximum Likelihood estimation (MLE): Choose value that maximizes the probability of observed data.
	$$\hat \theta_{MLE}=\underset{\theta}argmaxP(D\vert \theta)$$
	Maximum a posteriori(MAP) estimation: Choose value that is most probable given observed data and prior belief.
	$$\hat \theta_{MAP}=\underset{\theta}argmaxP(\theta\vert D)=\underset{\theta}argmaxP(D\vert \theta)*P(\theta)$$
	What we could conclude then, is that MLE is a special case of MAP, where the prior probability is uniform (the same everywhere)!

---
####  The exponential family:
##### References:
1. [https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf)
2. [www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf](www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf)
3. [https://ocw.mit.edu/courses/mathematics/18-655-mathematical-statistics-spring-2016/lecture-notes/MIT18_655S16_LecNote7.pdf](https://ocw.mit.edu/courses/mathematics/18-655-mathematical-statistics-spring-2016/lecture-notes/MIT18_655S16_LecNote7.pdf)

(Chinese reference) 

1. [http://blog.csdn.net/dream_angel_z/article/details/46288167](http://blog.csdn.net/dream_angel_z/article/details/46288167)
2. [http://www.cnblogs.com/huangshiyu13/p/6820729.html](http://www.cnblogs.com/huangshiyu13/p/6820729.html)

Given a measure η, we define an exponential family of probability distributions as those distributions whose density (relative to η) have the following general form: 

> $$ p(x\vert η) = h(x)e^{η^T . T(x) − A(η)} $$

>Key point: x and η only “mix” in $$exp^{(η^T . T(x))}$$

>η : vector of "nature parameters"
>
>T(x): vector of "Natural Sufficient Statistic"
>
>A(η): partition function / cumulant generating function

>h : X → R
>
>η : Θ → R
>
>B : Θ → R.

#### Generalized Linear Model, GLM
##### References:
1. [http://www.cs.princeton.edu/courses/archive/spr09/cos513/scribe/lecture11.pdf](http://www.cs.princeton.edu/courses/archive/spr09/cos513/scribe/lecture11.pdf)

  The generalized linear model (GLM) is a powerful generalization of linear regression to more general exponential family. The model is based on the following assumptions:

1. The observed input enters the model through a linear function $$(β^T X)$$.
2. The conditional mean of response, is represented as a function of the linear combination: $$E[Y\vert X]$$ is defined as $$µ = f(β^T.X)$$. 
3. The observed response is drawn from an exponential family distribution with conditional mean µ.

  η = Ψ(µ)
  
  where Ψ is a function which maps the natural (canonical) parameters to the mean parameter. µ defined as E[t(X)] can be computed from dA(η)/dη which is solely a function η.

 [ (xn)-->(yn)<--]--(β) (Representation of a generalized linear model)

 (β^T.X)--f(β^T.X)--> µ-- Ψ(µ)-->η (Relationship between the variables in a generalized  linear model)

---
####  Kullback-Leibler divergence (KL Divergence) / Information Gain / relative entropy

The KL divergence from  y^(or Q, your observation)  to  y (or P, ground truth)  is simply the difference between cross entropy and entropy:

$$ KL(y \vert\vert \hat{y})=\sum_iy_ilog\frac{1}{\hat{y}_i}−\sum_iy_ilog\frac{1}{y_i}=\sum_iy_ilog\frac{y_i}{\hat{y}_i} $$

In the context of machine learning, $$KL(P\vert\vert Q)$$ is often called the information gain achieved if Q is used instead of P. By analogy with information theory, it is also called the relative entropy of P with respect to Q.

---

#### Learning Theory & VC dimension(for Vapnik–Chervonenkis dimension)
##### References:
1. [https://drive.google.com/file/d/0B6pX3VvUVMAIeVk4OXlxRk0tcXM/view](https://drive.google.com/file/d/0B6pX3VvUVMAIeVk4OXlxRk0tcXM/view)
2. [https://www.cs.cmu.edu/~epxing/Class/10701/slides/lecture16-VC.pdf](https://www.cs.cmu.edu/~epxing/Class/10701/slides/lecture16-VC.pdf)
3. [http://cs229.stanford.edu/notes/cs229-notes4.pdf](http://cs229.stanford.edu/notes/cs229-notes4.pdf)

  Definition: The Vapnik-Chervonenkis dimension, VC(H), of hypothesis space Hdefined over instance space Xis the size of the largest finite subsetof Xshattered by H. If arbitrarily large finite sets of Xcan be shattered by H, then VC(H) is defined as infinite.

  VC dimension is a measure of the capacity (complexity, expressive power, richness, or flexibility) of a space of functions that can be learned by a statistical classification algorithm. It is defined as the cardinality of the largest set of points that the algorithm can shatter.

---

#### Statistical forecasting
##### ARIMA (Auto-Regressive Integrated Moving Average)
  1. A series which needs to be differenced to be made stationary is an “integrated” (I) series
  2. Lags of the stationarized series are called “autoregressive” (AR) terms
  3. Lags of the forecast errors are called “moving average” (MA) terms
  4. Non-seasonal ARIMA model “ARIMA(p,d,q)” model
    . p = the number of autoregressive terms
    . d = the number of nonseasonal differences
    . q = the number of moving-average terms
  5. Seasonal ARIMA models, “ARIMA(p,d,q)X(P,D,Q)” model
    . P = # of seasonal autoregressive terms
    . D = # of seasonal differences
    . Q = # of seasonal moving-average terms
  6. Augmented Dickey-Fuller (ADF) test of data stationarity
    If test statistic < test critical value %1 => Data is stationarity.
  7. Data stationarity
    1. The mean of the series should not be a function of time. 
    2. The variance of the series should not be a function of time.
    3. The covariance of the i th term and the (i + m) th term should not be a function of time.
  8. Transformations to stationarize the data.
    1. Deflation by CPI
    2. Logarithmic
    3. First Difference
    4. Seasonal Difference
    5. Seasonal Adjustment


Reference: 
[http://people.duke.edu/~rnau/411home.htm](http://people.duke.edu/~rnau/411home.htm)



## Disclaimer
Last updated: March 13, 2018

The information contained on https://github.com/Cheng-Lin-Li/ website (the "Service") is for general information purposes only.
Cheng-Lin-Li's github assumes no responsibility for errors or omissions in the contents on the Service and Programs.

In no event shall Cheng-Lin-Li's github be liable for any special, direct, indirect, consequential, or incidental damages or any damages whatsoever, whether in an action of contract, negligence or other tort, arising out of or in connection with the use of the Service or the contents of the Service. Cheng-Lin-Li's github reserves the right to make additions, deletions, or modification to the contents on the Service at any time without prior notice.

### External links disclaimer

https://github.com/Cheng-Lin-Li/ website may contain links to external websites that are not provided or maintained by or in any way affiliated with Cheng-Lin-Li's github.

Please note that the Cheng-Lin-Li's github does not guarantee the accuracy, relevance, timeliness, or completeness of any information on these external websites.

## Contact Information

mailto:[clark.cl.li@gmail.com](mailto:clark.cl.li@gmail.com)
