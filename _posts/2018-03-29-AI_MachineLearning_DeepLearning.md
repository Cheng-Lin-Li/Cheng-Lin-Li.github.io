---
layout: post
title: What are differences between AI, Machine Learning, and Deep Learning?
tags: concept ai machine-learning deep-learning
---

![AI, machine learning, deep learning, and others](/images/2018-03-29-AI.svg)

Deep learning ⊂ Machine learning ⊂ Artificial Intelligence 

In one sentence, deep learning is a subset of machine learning in Artificial Intelligence (AI).

<!-- more -->

---
Table of content:

{: class="table-of-content"}
* TOC
{:toc}

---

## Artificial Intelligence (AI)

### What is Artificial Intelligence ?

According to the definition of Association for the Advancement of Artificial Intelligence (AAAI):

“The scientific understanding of the mechanisms underlying thought and intelligent behavior and their embodiment in machines.”

In another word, A.I. attempts to build intelligent entities to perceive, understand, predict, and manipulate a world far larger and more complicated than itself.

Traditionally, four approaches to define A.I.:
1. Acting humanly: The Turing Test approach
    You can find what is [the Turing Test by wiki](https://en.wikipedia.org/wiki/Turing_test)The major target is to let computer possess the following capabilities:
    a. natural language processing
    b. knowledge representation
    c. automated reasoning
    d. machine learning.
2. Thinking humanly: The cognitive modeling approach
    From determine how humans think? The cognitive science brings together computer models from AI and experimental techniques from psychology to construct precise and testable theories of the human mind.
3. Thinking rationally: The “laws of thought” approach
    From the Greek philosopher Aristotle's [syllogisms](https://en.wikipedia.org/wiki/Syllogism) to the field of logic, this approach would like to develop irrefutable reasoning process and hope programs could, in principle, solve any solvable problem described in logical notation. In the “laws of thought” approach to AI, the emphasis was on correct inferences.

    There are two main obstacles to this approach. First, it is not easy to take informal knowledge and state it in the formal terms required by logical notation. Second, there is a big difference between solving a problem “in principle” and solving it in practice.
4. Acting rationally: The rational agent approach, the mainstream approach as of today.
    An agent is just something that acts autonomously, perceive their environment, persist over a prolonged time period, adapt to change, and create and pursue goals to achieve the best outcome or, when there is uncertainty, the best expected outcome.

    The rational-agent approach has two advantages over the other approaches. First, it is more general than the “laws of thought” approach because correct inference is just one of several possible mechanisms for achieving rationality. Second, it is more amenable to scientific development than are approaches based on human behavior or human thought. The standard of rationality is mathematically well defined and completely general, and can be “unpacked” to generate agent designs that provably achieve it.

    One important thing is that achieving perfect rationality—always doing the right thing—may be not feasible in complicated environments due to long time computations, the issue of limited rationality—acting appropriately when there is not enough time to do all the computations should be considered in this approach.

Below areas are foundations of A.I.:

1. Philosophy
2. Mathematics
3. Economics
4. Neuroscience
5. Psychology
6. Computer engineering
7. Control theory and cybernetics
8. Linguistics

## Machine Learning

### Machine learning is a subset of AI 

Machine Learning gives an agent the ability to "learn" (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed. An agent is learning if it improves its performance on future tasks after making observations about the world.

#### Two categories to aggregate machine learning algorithms:

1. Deductive learning: From Generalized rules to lead to correct knowledge.
    Pros:
        1. Logical inference generates entailed statements.
        2. Probabilistic reasoning can lead to updated belief states.
    Cons:
        1. We often have insufficient knowledge for inference.

2. Inductive learning: From examples or activities to lead to generalized rules. It may arrive at incorrect conclusions.
    Pros:
        1. The learning can be better than not trying to learn at all.
    Cons:
        1. Local optimals rather than global optimal.
        2. Overfitting issues.

#### Three types of machine learning algorithms:
If you want to split machine learning into three types types of learning, that will be supervised learning, unsupervised learning, and reinforcement learning. 

1. Supervised Learning: Algorithms find a solution based on labeled sample data.

    All regression tasks and most of classifications rely on labeled data to feed into algorithms. For example, machine can classify cat, dog, car, ... from an image after trained by thousands of labeled images.

2. Unsupervised Learning: Algorithms give an answer based on unlabeled data.

    How algorithm can learn knowledge from "unlabeled" data? Basically this kind of algorithms perform classification tasks which clusters data into different sets(or groups) based on "distance" of features from each data. Clustering and Dimensionality reduction algorithms rely on unsupervised learning.

    Is unsupervised learning useful? Of course it is. For instance, you search from google by some key words to get tones of websites is based on unsupervised learning algorithm.

3. Reinforcement Learning: Algorithms based on long-term rewards to learn the rules/answers.

    You may not know reinforcement learning approach, but you definitely heart about AlphGo beats human champaign in 2017. Yes, it is reinforcement learning explore new approaches to play games.

    Given reward functions and the environment’s states, the agent will choose the action to maximize rewards or explore new possibilities.


## Deep Learning

### Deep learning is a subset of machine learning.

Deep learning focus on multiple (deep) layer of Artificial Neural Network(ANN) with different combinations. This kind of algorithms dominate computer vision, sound recognition, machine translations...etc.

Scientists construct different architectures of connections between different activation functions (to simulate the behavior of neurons) which actually project data from original question space to a new solution space to solve it (find the answer).

These algorithm split all inputs (image, voice, text) into high dimensional matrix of numbers to compute them. Those matrix operations will take a lot of time and computing powers on CPU. Because the powerful graphics processing unit (GPU) was developed in recently years, it helps deep learning to be practicable and reveal the power of these algorithms.

The research competition in this area is not only related to algorithm design but also computing power. You don't want to wait for 1 weeks to see the result of experiments. That's why GPU card (or high-end graphic card) is very important for machine learning researchers today.

I don't have a GPU card, how can I do for deep learning research? The good news is Google provides a free (so far) GPU developing enviroment call [Google Colaboratory] for you with some limitations. You may [click here to try it](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) or [click here for more detail](https://research.google.com/colaboratory/faq.html#browsers).


## Reference:
1. Stuart Russel, Peter Norvig, Artificial Intelligence - A Mordern Approach, Third Edition.


Revised on April, 1, 2018 for machine learning and deep learning.
Revised on April, 30, 2018 to include table of content.