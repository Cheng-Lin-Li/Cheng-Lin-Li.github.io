---
layout: post
title: What is the difference between AI, Machine Learning, and Deep Learning?
---

![AI, machine learning, deep learning, and others](/images/2018-03-29-AI.svg)
### Artificial Intelligence (A.I.)

What is Artificial Intelligence ?

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

From determine how humans think? The cognitive science brings together
computer models from AI and experimental techniques from psychology to construct precise and testable theories of the human mind.

3. Thinking rationally: The “laws of thought” approach

From the Greek philosopher Aristotle's [syllogisms](https://en.wikipedia.org/wiki/Syllogism) to the field of logic, this approach would like to develop irrefutable reasoning process and hope programs could, in principle, solve any solvable problem described in logical notation. In the “laws of thought” approach to AI, the emphasis was on correct inferences.

There are two main obstacles to this approach. First, it is not easy to take informal knowledge and state it in the formal terms required by logical notation. Second, there is a big difference between solving a problem “in principle” and solving it in practice.

4. Acting rationally: The rational agent approach, the mainstream approach as of today.

An agent is just something that acts autonomously, perceive their environment, persist over a prolonged time period, adapt to change, and create and pursue goals to achieve the
best outcome or, when there is uncertainty, the best expected outcome.

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

### Machine Learning

Machine learning is a subset of AI that gives an agent the ability to "learn" (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed. An agent is learning if it improves its performance on future tasks after making observations
about the world.

Two categories to aggregate machine learning algorithms:

Deductive learning: From Generalized rules to lead to correct knowledge.
    Pros:
        1. Logical inference generates entailed statements.
        2. Probabilistic reasoning can lead to updated belief states.
    Cons:
        1. We often have insufficient knowledge for inference.
    Methods: Traditional or Statistics algorithms. For example:
        1. First order logic
        2. Bayesian theory.

Inductive learning: From examples or activities to lead to generalized rules. It may arrive at incorrect conclusions.
    Pros:
        1. The learning can be better than not trying to learn at all.
    Cons:
        1. Local optimals rather than global optimal.
        2. Overfitting issues.
    Methods:
        1. Artificial Neural Network



### Deep Learning


Reference:
1. Stuart Russel, Peter Norvig, Artificial Intelligence - A Mordern Approach, Third Edition.
2. 