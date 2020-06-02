# Meta-Learning

---
## Motivation

- Humans do not learn from scratch anytime!
- Fewer samples are needed to master a new skill.
- **Goal:** Can we have a machine do the same?
- Very important for deep learning systems.
![](../img/data.png)

---
## Human vs Machine

- **Sample efficiency:** Humans need few samples, even relatively simple systems like handwritten digit recognition need thousands of samples.
- **Transferability:** Less samples required, but still critical parts or novel components might be hard to tweak.

---
## Meta-Learning: Learning to learn
- A model is trained over a variety of tasks. 
- Each task is associated with a dataset that contains input features and a target variable (supervised learning). 
- The **model** on a meta-learning problem is a high-level optimizer that updates a low-level model, which is specialized for the task.


---
## Example: 4-shot 2-class image classification
![](../img/fewshot.png)


---
## Common approaches
- **Model-based:** Similar to RNN.
- **Metric-based:** Learn embedding and distance function to separate classes.
- **Optimization-based:** Gradient descent.

---
## Metric-Based Meta-Learning: Siamese networks
- Twin networks that act as feature extractors.
- The discriminator calculates the distance between embeddings and issues a class probability.
![](../img/siamese.png)

---
## Metric-Based Meta-Learning: Matching network
![](../img/matching.png)

---
## Metric-Based Meta-Learning: Relation network
![](../img/relation.png)

---
## Model-Based Meta-Learning: MANN
![](../img/mann.png)
- Present data to the network with time-offset labels.
- Focus of the training is on a) learning the embedding, b) retrieving on memory.



---
## Optimization-Based
- **Idea:** We love gradients. Can we make them work in this few-data setting too?
![](../img/maml.png)
---
## Optimization-Based: MAML (model-agnostic)

![](../img/maml-algo.png)
- [Summary article from the authors.](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)


---
## Optimization-Based: MAML (cont.)
![](../img/maml-pic.png)
- **Support set:** a classification task on which we will train.

---
## Optimization-Based: Reptile
![](../img/reptile.png)

---
## Beyond MAML: ANIL (Almost no inner loop)
![](../img/anil.png)
- Same as in MAML, but take the gradient only of the head of the model, keeping the feature extractor part.

---
## This is all very nice, but where is AutoML?
![](../img/autosklearn.png)
- Meta-learning helps determine which algorithm to use.
	- Based on 140 datasets from OpenML repository and meta-features (statistics, skewness, entropy of the targets).
- BO then kicks in to solve the HPO part.
- Create an ensemble model in the end.

---
## Auto-sklearn: results
![](../img/autosklearn-results.png)

---
## Auto-sklearn vs sklearn
![](../img/autosklearn-results2.png)


---
## References
- [Meta-learning survey (mostly deep learning)](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html).
- [MANN original article](http://proceedings.mlr.press/v48/santoro16.pdf)
- [Learning to learn (MAML)](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/).
- [Reptile, with implementation](https://openai.com/blog/reptile/).
- [Meta-Learning with Python book, with code](https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python)
- [Auto-sklearn paper](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)