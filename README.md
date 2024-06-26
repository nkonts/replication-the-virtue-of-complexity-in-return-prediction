# Replication: The Virtue of Complexity in Return Prediction
This repo is replicating the following paper:

**Kelly, Bryan T. and Malamud, Semyon and Zhou, Kangying**, *The Virtue of Complexity in Return Prediction*, December 2021, Swiss Finance Institute Research Paper No. 21-90. Available at SSRN: https://ssrn.com/abstract=3984925 or http://dx.doi.org/10.2139/ssrn.3984925.

> Contrary to conventional wisdom in finance, return prediction R2 and optimal portfolio Sharpe ratio generally increase with model parameterization, even when minimal regularization is used. We theoretically characterize the behavior of return prediction models in the high complexity regime, i.e. when the number of parameters exceeds the number of observations. Empirically, we document this "virtue of complexity" in US equity market prediction. High complexity models deliver economically large and statistically significant out-of-sample portfolio gains relative to simpler models, due in large part to their remarkable ability to predict recessions.

The dataset referenced in the paper is available at Amit Goyal's [webpage](https://sites.google.com/view/agoyal145/) and from his paper [A Comprehensive Look at the Empirical Performance of Equity Premium Prediction](https://drive.google.com/file/d/1uvjBJ9D09T0_sp7kQppWpD-xelJ0KQhc/view?usp=sharing) (with Ivo Welch), July 2008, Review of Financial Studies 21(4) 1455‒1508.

---

**experiment.ipynb**: This notebook serves as a testing ground for the initial setup and validation of the model. It's where the dataset was replicated and a preliminary check was conducted to ensure that everything was functioning as expected. In other words, it's a single run-through of the model, used for quick prototyping and troubleshooting.

**simulation.ipynb**: This notebook delves deeper into the process and is designed for more extensive replication. The code has been modularized for better organization and readability. This notebook executes the experiment in *500 iterations* to minimize the impact of variance introduced by the randomly generated RFF features. It's designed for more thorough, comprehensive, and reliable testing or simulation.

---

# Notes about the model - Ridge Regression with Moore-Penrose pseudo-inverse:

This is the formula from the paper. It has a run time of ~6mins for me and we will calculate *many* iterations where this will be used. The total runtime becomes unfeasable for my home computer.

> ```beta = np.linalg.pinv(z*np.identity(nr_cols) +  (S.T @ S)/T) @ (R @ S)/T```

Doing it with the normal inverse instead of the pseudo inverse has a run time of 25sec (not numerically stable if not full rank, numbers can get incredibly large).

>```beta = np.linalg.inv(z*np.identity(nr_cols) +  (S.T @ S)/T) @ (R @ S)/T```

*Note: I did not observe different results with inv() vs. pinv().*

This formula (notice that the division by `T` is missing) is equivalent to the sklearn ridge regression but orders of magnitude slower compared to sklearn (*numpy: 6 mins, Sklearn: 45 ms*)
>```beta = np.linalg.pinv(z*np.identity(nr_cols) +  (S.T @ S)) @ (R @ S)```

We can adjust Ridge.alpha with `T` to get full equivalence between a manual numpy calculation and the more optimized code from sklearn:

```
   np.linalg.pinv(z*np.identity(nr_cols) +  (S.T @ S)/T) @ (R @ S)/T
== np.linalg.pinv(z*np.identity(nr_cols)*T/T +  (S.T @ S)/T) @ (R @ S)/T
== np.linalg.pinv(T*z*np.identity(nr_cols) +  (S.T @ S))*T @ (R @ S)/T
== np.linalg.pinv(T*z*np.identity(nr_cols) +  (S.T @ S)) @ (R @ S)
```

Since the Sklearn solution is overall 8.000 times faster compared to the original formula, I will proceed with that solution. 
I did only test ~100 iterations in the comparision between `pinv()`, `inv()` and `Ridge()` to numerically check if this approach results in the same output.

# Result:

## Different metrics for a 1-step ahead prediction with 12 data points and gamma==2 and 0.5:
The baseline regression is represented by a grey horizontal line. 
### Gamma == 2
![Metrics](plots/metrics_2.jpg "Metrics")
### Gamma == 0.5
![Metrics](plots/metrics_0.5.jpg "Metrics")

## Sharpe Ratio for different gamma:
The baseline regression is indicated by a grey horizontal line.  

![Sharpe Ratio by Gamma](plots/gamma_SR.jpg "Sharpe Ratio by Gamma")

## Market Timing positions vs NBER Recessions:
![Forecast vs NBER Recessions](plots/result_forecast.jpg "Forecast vs NBER Recessions") 
### Expected plot:
![Expected plot](plots/expected_result.png "Expected plot")
