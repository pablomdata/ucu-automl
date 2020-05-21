# Exercise 1

- We will work with the `diabetes` data in scikit-learn (via the `load_diabetes`) function.
- Compare three methods for hyperparameter optimization for `RandomForestRegressor` with hyperparameters `max_depth`, `n_estimators`, `max_features`.
	- *Hint:* You should write a function that takes a 3-dimensional array and returns a scalar /1-d arrray.
- To compare the methods, plot the number of iterations against the loss (negative MSE). 
- Methods to compare: 	
	- Random Search (`RandomizedSearchCV`).
	- Grid Search (`GridSearchCV`).
	- Bayesian Optimization using GPy/GPyOpt.
	- Extra: Another black-box method (i.e. cross-entropy method, genetic algorithm, simulated annealing).
	
