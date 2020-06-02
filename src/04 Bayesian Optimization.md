# Bayesian Optimization

---
## Motivation
[Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/)

---
## HPO problem as a Bayesian Optimization problem
1. **Objective function:** what we want to minimize
2. **Configuration space:** possible values of the hyperparameters.
3. **Surrogate function:** a model for $p(y | x)$. For HPO, $y$ represents the **loss** and $x$ the **configuration**.
4. **Trials:** score, parameter pairs recorded each time we evaluate the objective function.


---
## Sequential Model-Based Optimization
- After each evaluation, the probability model gets updated.
- Next values to try are selected by the algorithm according to a criteria, usually Expected Improvement.
- Finding values that maximize expected improvement is cheaper than evaluating the function itself.
- Having a probabilistic model gives us hope that convergence will take less time.

---
## Sequential Model-Based Optimization (cont.)

There are different choices for building the surrogate model.
- **Gaussian Processes:** $p(y | x) \approx \mathcal N(\mu_K, \sigma_K)$
	- $K$ is a *kernel function* that is used to calculate a local mean and variance. 
- **Random Forest Regression:** $p(y | x) \approx \mathcal N(\mu_B, \sigma_B)$
	- $\mu_B, \sigma_B$ are calculated over the values given by a regression forest.
- **Tree-structured Parzen Estimator**

---
## Tree Parzen Estimator
- Instead of modeling $p(y|x)$, one models $p(x|y)$ and $p(y)$ directly.
- This is achieved by estimating two different processes $\ell(x)$ and $g(x)$, each of which is estimated from quantiles of $y$.

---
## Implementations

- [SMAC (Random forests)](https://automl.github.io/SMAC3/master/quickstart.html)
- [TPE](http://hyperopt.github.io/hyperopt/)
- Gaussian Processes: [GPyOpt](https://sheffieldml.github.io/GPyOpt/), 
[scikit-optimize](https://scikit-optimize.github.io/stable/).
	- For classification and regression: [scikit-learn](https://scikit-learn.org/stable/modules/gaussian_process.html)
---
## References
- [Bayesian Optimization Primer](https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf)
- [Hyperopt Jupyter Notebook example, including GBM example.](https://github.com/WillKoehrsen/hyperparameter-optimization)
- [Practical Bayesian Optimization of ML algorithms](https://arxiv.org/pdf/1206.2944.pdf)
- [Algorithms for hyperparameter optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)