# AutoML libraries

---
## Summary (2020)

| System | API | Algorithms | Meta-learners |
|--------|-----|------------|-------|
| Auto-WEKA | Java | Classifiers in WEKA. | SMAC. |
| Hyperopt-sklearn | Python | Some classifiers from sklearn.| TPE, Annealing, GP. 
| Auto-sklearn| Python | Classif. / reg. | Meta-learning, BO, SMAC, ensembling. | 
| AutoGluon | Python (Linux/Mac) | Tabular / image algos | BO, Hyperband.
| Optunity | Python, R, Matlab | Classif. / reg. |PSO, CMA-ES, Sobol sequences, simplex.|

---
## What about H2O?
- H2O's focus is on **big data**: native parallelization and GPU usage, but hyperparameter selection is with *random search*.
- There is an ensembling step as in `auto-sklearn`, but it is simple model stacking, not model selection.
- Detailed comparison [here](https://arxiv.org/pdf/1908.05557.pdf).
---
## Links
- [Hyperopt-sklearn](http://hyperopt.github.io/hyperopt-sklearn/)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/master/)
- [Optunity](https://optunity.readthedocs.io/en/latest/)
- [TPOT, in development](http://epistasislab.github.io/tpot/)

**Deep Learning**
- [Gluon, AutoML for Deep Learning](https://autogluon.mxnet.io/tutorials/index.html)
- [Auto-keras](https://arxiv.org/pdf/1908.05557.pdf)

**Pure meta-learning**
- [Learn2Learn](http://learn2learn.net/)
	- Experimental, as of May 2020, not production-ready.

