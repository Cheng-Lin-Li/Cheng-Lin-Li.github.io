---
layout: post
title: Hello Data !!
---

This is my learning notebook which includes AI, Machine Learning, Big Data techniques, Knowledge Graph, Information Visualization and Natural Language Processing.

I am a lifelong learner and passinate to contribute my knowledge to impact the world. After twenty years dedicate working on IT Application development departments for Intranet, B2C, B2B eCommerce portal and trading Websites, I observed the AI and DATA era is coming. It is time to let DATA tell its story by AI / machine learning algorithms and that's the reason why I resigned from J.P. Morgan Asset Management (Taiwan) in 2016 and went back to school to be a graduate student in Viterbi school of engineering at University of Southern California. My research area is data informatics and I would like to share what I learn with everyone.

There are several GitHub repositories below for your reference as first step. I hope my works can help you to understand all those concepts and algorithms:

[ >>AI<< ](https://cheng-lin-li.github.io/AI/) includes my implementations of classical A.I. algorithms, like Alpha-Beta Pruning, Propositional Logic, and decision networks (Not yet complete).

[ >>Machine Learning<< ](https://cheng-lin-li.github.io/MachineLearning/) includes my implementations of machine learning algorithms, programs demostrate how to use scikit-learn, and tensorflow frameworks.

[ >>Spark<< ](https://cheng-lin-li.github.io/Spark/) include A-priori and SON, ALS with UV Decomposition, TF-IDF with K-Means, Matrix Multiplication by Two Phases approach, and Minhash and Locality-Sensitive Hash (LSH).

[ >>Information Visualization<< ](https://cheng-lin-li.github.io/InformationVisualization/) includes my codes for D3js, R, ggplot2.

[ >>Knowledge Graph<< ](https://cheng-lin-li.github.io/KnowledgeGraph/) include utilities programs like JSONLines package program, facebook crawler, information extraction, and a final project to combine all those techniques and machine learning algorithm to predict the trend of Dow Jones Industrial Average (DJIA) in next day and next 30 day.

[ >>Neural Language Processing<< ](https://cheng-lin-li.github.io/Natural-Language-Processing/) include tagging program based on HMM-Viterbi with add one smoothing for transition probability, Good Turing smoothing / rule base algorithm for unknown words. (under constructing).


---

Projects and Competition notebooks:

  1. Stock Price Forecasting by Stock Selections: Python/Tensorflow
    
      This is a project which implemented Neural Network and Long Short Term Memory (LSTM) for stock price predictions. These models beat DJIA performance based on 1 quarter of weekly price, return rate of the DJIA components plus assistant indices to predict the highest increasing rate stock for the next quarter.

      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/TensorFlow)

  2. Market Trend Prediction with Social Media Listening: Python/Keras/Facebook Graph API/Twitter API/ACHE Crawler/Beautifulsoup
      
      This project leveraged 1.5 years of 30 historical stock prices, Dow Jones Industrial Average(DJIA) index, with semantic information from social media (Facebook and Twitter) on T day to provide better one to many DJIA trend classifications for T+1/T+30 days than the model without social media info by LSTM in python a Keras.

      Github address: [https://github.com/Cheng-Lin-Li/Market-Trend-Prediction](https://github.com/Cheng-Lin-Li/Market-Trend-Prediction)
  3. Information Visualization Project - Business Cycle Introduction
      
      Build up a web application to introduce what business cycle is and how it will impact to us.

      Github address: [https://github.com/Cheng-Lin-Li/InformationVisualization/tree/master/BusinessCycle](https://github.com/Cheng-Lin-Li/InformationVisualization/tree/master/BusinessCycle)
  4. Number of vehicles Prediction: scikit-learn/Keras
      This task is to perform prediction for number of vehicles by given data. This is a demo program to leverage four models (SVR, NN, LSTM, GRU) from existing libraries in one challenge. The final result can be improved by some emsemble techniques like Bootstrap aggregating (bagging), boosting, and stacking to get better performance.
      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/McKinseyAnalyticsPrediction)
  5. Recommendation System: scikit-learn/Surprise (under constructing)
      This task leverages Content Based Filtering and Singular Value decomposition (SVD) to perform recommendation system build up.

      Github address: [https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsRecommendation/Recommendation.ipynb](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/McKinseyAnalyticsRecommendation/Recommendation.ipynb)


  6. Objects detection and segmentation: Keras/Tensorflow/OpenCV(under constructing)
      This task is based on Mask RCNN (extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition) to perform transfer learning on Nuclear detection from varience image files.

        Reference Paper: [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)

        The original model clone from: [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)

        Github address:[https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/ObjectDetectionSegmentation](https://github.com/Cheng-Lin-Li/MachineLearning/tree/master/Competition/ObjectDetectionSegmentation)

        My trial work to integrate threading webcam stream and the pre-trained for object detection and segmentation tasks.
	Github address:[https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/ObjectDetectionSegmentation/Video-Demo-Mask_RCNN.ipynb](https://github.com/Cheng-Lin-Li/MachineLearning/blob/master/Competition/ObjectDetectionSegmentation/Video-Demo-Mask_RCNN.ipynb)


I will leverage my spare time to enrich this notebook from time to time. Your comments are appreciated.

You can contact me via emails.

### Some Technology notes are drafted here, I may write articles for them in the future.

---
### Foundations
Mathematics is the foundation of science, you may need to review below areas if you have any difficulty on rest of topics.

1. Calculus
2. Statistic
  2-1. Probabilities and Expectations

    2-1-1. [ReviewofProbabilityTheory at Stanford CS229 machine learning](http://cs229.stanford.edu/section/cs229-prob.pdf)
  
    2-1-2. Distributions and Tests
  
3. Linar Algebra / Discrete Mathematics

    3-1 [Linear Algebra Review and Reference at Stanford CS229 machine learning](http://cs229.stanford.edu/section/cs229-linalg.pdf)


---

### Artificial intelligence (AI)

Textbook:

1. [Stuart Russell,‎ Peter Norvig, Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)

---

### Machine Learning (ML)

Articles & Papers:
1. [Deep Learning: An Introduction for Applied Mathematicians](https://arxiv.org/abs/1801.05894v1)
2. [The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches](https://arxiv.org/abs/1803.01164)

Training Material and Courses:
1. [Representation learning in Montreal Institute for Learning Algorithms @Universite' de Montre'al](https://ift6135h18.wordpress.com/) (Deep Learning)
2. [Reinforcement Learning: An Introduction by Prof. Richard S. Sutton & Andrew G. Barto @University of Alberta](https://drive.google.com/drive/folders/0B3w765rOKuKANmxNbXdwaE1YU1k), [OR try this alternative link.](http://www.incompleteideas.net/book/the-book-2nd.html)
3. [Carnegie Mellon University - 10715 Advanced Introduction to Machine Learning: lectures](https://sites.google.com/site/10715advancedmlintro2017f/lectures)

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
#### What is Artificial Intelligence ?
Association for the Advancement of Artificial Intelligence (AAAI): 

“The scientific understanding of the mechanisms underlying thought and intelligent behavior and their embodiment in machines.”

There are different categories to aggretate AI algorithms:

* Deductive learning: Leads to correct knowledge.
  - Pros:
    + Logical inference generates entailed statements.
    + Probabilistic reasoning can lead to updated belief states.
  - Cons:
    + We often have insufficient knowledge for inference.
  - Methods:
    Classical AI algorithms.
* Inductive learning: Arrives at may be incorrect conclusions.
  - Pros:
     + The learning can be better than not trying to learn at all.
  - Cons: 
     + Local optimals rather than global optimal.
     + Overfitting issues.
  - Methods:
    Machine learning algorithms.
     
     
---
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
1. [http://cs229.stanford.edu/notes/cs229-notes2.pdf](http://cs229.stanford.edu/notes/cs229-notes2.pdf)


* Generative model

  Generative learning algorithms. For instance, if y indicates whether an example is a dog (0) or an elephant (1), then p(x\|y = 0) models the distribution of dogs’ features, and p(x\|y = 1) models the distribution of elephants’ features.

  1. Gaussian mixture model and other types of mixture model
  2. Hidden Markov model
  3. Probabilistic context-free grammar
  4. Naive Bayes
  5. Averaged one-dependence estimators
  6. Latent Dirichlet allocation
  7. Restricted Boltzmann machine
  8. Generative adversarial networks

* Discriminative model, also called conditional models.

  Algorithms that try to learn p(y\|x) directly (such as logistic regression), or algorithms that try to learn mappings directly from the space of inputs X to the labels {0,1}, (such as the perceptron algorithm) are called discriminative learning algorithms. 

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

Formular: Var(W) = 1/n_in

where W is the initialization distribution for the neuron in question, and n_in is the number of neurons feeding into it. The distribution used is typically Gaussian or uniform.

It’s worth mentioning that Glorot & Bengio’s paper originally recommended using: 
	Var(W) = 2/(n_in+n_out)
	where n_out is the number of neurons the result is fed to.

####  He initialization: For the more recent rectifying nonlinearities (ReLu)
##### References:
1. [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)

Formular: Var(W) = 2/n_in

Which makes sense: a rectifying linear unit is zero for half of its input, so you need to double the size of weight variance to keep the signal’s variance constant.

---
####  GloVe: Global Vectors for Word Representation
##### References:
1. [Jeffrey Pennington, Richard Socher, Christopher D. Manning, GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

	GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

___
####  F_(beta) score: An easy to combine precision and recall measures
>
>F_(beta) = (1+beta^2)(precision*recall)/(beta^2*precision+recall)
>
>F1 = 2*(precision*recall)/(precision+recall)

___
#### Symmetric Mean Absolute Percent Error (SMAPE)
##### References:
1. [http://www.vanguardsw.com/business-forecasting-101/symmetric-mean-absolute-percent-error-smape/](http://www.vanguardsw.com/business-forecasting-101/symmetric-mean-absolute-percent-error-smape/)

	An alternative to Mean Absolute Percent Error (MAPE) when there are zero or near-zero demand for items. SMAPE self-limits to an error rate of 200%, reducing the influence of these low volume items. Low volume items are problematic because they could otherwise have infinitely high error rates that skew the overall error rate.
	SMAPE is the forecast minus actuals divided by the sum of forecasts and actuals as expressed in formula:
  

>SMAPE = 2/N * Sum_(k=1~N)\|Fk-Ak\|/(Fk+Ak)
>
>k = each time period.

####  Mean Absolute Percent Error (MAPE)
##### References:
1. [http://www.vanguardsw.com/business-forecasting-101/mean-absolute-percent-error/](http://www.vanguardsw.com/business-forecasting-101/mean-absolute-percent-error/)

Mean Absolute Percent Error (MAPE) is the most common measure of forecast error. MAPE functions best when there are no extremes to the data (including zeros).

With zeros or near-zeros, MAPE can give a distorted picture of error. The error on a near-zero item can be infinitely high, causing a distortion to the overall error rate when it is averaged in. For forecasts of items that are near or at zero volume, Symmetric Mean Absolute Percent Error (SMAPE) is a better measure.
	MAPE is the average absolute percent error for each time period or forecast minus actuals divided by actuals:

>MAPE = 1/N * Sum_(k=1~N)\|Fk-Ak\|/Ak
>
>k = each time period.

---

####  MLE vs MAP: the connection between Maximum Likelihood and Maximum A Posteriori Estimation
##### References:
1. [http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/MLE_MAP_Part1.pdf](http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/MLE_MAP_Part1.pdf)
2. [https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/](https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/)
	
  Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP), are both a method for estimating some variable in the setting of probability distributions or graphical models. They are similar, as they compute a single estimate, instead of a full distribution.
	Maximum Likelihood estimation (MLE): Choose value that maximizes the probability of observed data.
	Cap Theta_MLE=argmax_(Theta)P(D|Theta)
	Maximum a posteriori(MAP) estimation: Choose value that is most probable given observed data and prior belief.
	Cap Theta_MAP=argmax_(Theta)P(Theta|D)=argmax_(Theta)P(D|Theta)*P(Theta)
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

Given a measure η, we deﬁne an exponential family of probability distributions as those distributions whose density (relative to η) have the following general form: 

>p(x|η) = h(x)exp{ηT . T(x) − A(η)}

>Key point: x and η only “mix” in exp(ηT . T(x))

>η : vector of "nature parameters"
>
>T(x): vector of "Natural Sufficient Statistic"
>
>A(η): partition function/ cumulant generating function

>h : X → R
>
>η : Θ → R
>
>B : Θ → R.

#### Generalized Linear Model, GLM
##### References:
1. [http://www.cs.princeton.edu/courses/archive/spr09/cos513/scribe/lecture11.pdf](http://www.cs.princeton.edu/courses/archive/spr09/cos513/scribe/lecture11.pdf)

  The generalized linear model (GLM) is a powerful generalization of linear regression to more general exponential family. The model is based on the following assumptions:

1. The observed input enters the model through a linear function (β^T X).
2. The conditional mean of response, is represented as a function of the linear combination: E[Y\|X] is defined as µ = f(β^T.X). 
3. The observed response is drawn from an exponential family distribution with conditional mean µ.

  η = Ψ(µ)
  
  where Ψ is a function which maps the natural (canonical) parameters to the mean parameter. µ deﬁned as E[t(X)] can be computed from dA(η)/dη which is solely a function η.

 [ (xn)-->(yn)<--]--(β) (Representation of a generalized linear model)

 (β^T.X)--f(β^T.X)--> µ-- Ψ(µ)-->η (Relationship between the variables in a generalized  linear model)

---
####  Kullback-Leibler divergence / Information Gain / relative entropy

In the context of machine learning, DKL(P\|\|Q) is often called the information gain achieved if Q is used instead of P. By analogy with information theory, it is also called the relative entropy of P with respect to Q.

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



### Disclaimer
Last updated: March 13, 2018

The information contained on https://github.com/Cheng-Lin-Li/ website (the "Service") is for general information purposes only.
Cheng-Lin-Li's github assumes no responsibility for errors or omissions in the contents on the Service and Programs.

In no event shall Cheng-Lin-Li's github be liable for any special, direct, indirect, consequential, or incidental damages or any damages whatsoever, whether in an action of contract, negligence or other tort, arising out of or in connection with the use of the Service or the contents of the Service. Cheng-Lin-Li's github reserves the right to make additions, deletions, or modification to the contents on the Service at any time without prior notice.

#### External links disclaimer

https://github.com/Cheng-Lin-Li/ website may contain links to external websites that are not provided or maintained by or in any way affiliated with Cheng-Lin-Li's github.

Please note that the Cheng-Lin-Li's github does not guarantee the accuracy, relevance, timeliness, or completeness of any information on these external websites.

## Contact Information
Cheng-Lin Li@University of Southern California

mailto: [chenglil@usc.edu](mailto:chenglil@usc.edu) , or [clark.cl.li@gmail.com](mailto:clark.cl.li@gmail.com)
