# Meshy Optimization

We consider the statistical problem where we measure a response $$y_i$$, and a covariate $x_i\in[0,1]$, on each of $i=1,\ldots,n$ observations. We assume a generative model of the form $$y_i = f_0 \left(x_i \right) + w_i$$, where $$f_0$$ is an unknown function from a known function class $$\mathcal{F}$$, and $w_i$ are iid errors with $\operatorname{E}\left[w_i\right] = 0$ and $\operatorname{var}{w_i} = \sigma^2 < \infty$. We are interested in estimating $f_0$ based on the observed data. One common approach for estimating $f$ is to use penalized regression:
	\begin{equation}\label{eq:pen}
	\hat{f} = \operatorname{argmin}_{f\in\mathcal{F}}\frac{1}{n}\sum_{i=1}^n\left(y_i - f\left(x_{i,\cdot}\right)\right)^2 + \lambda_n P\left(f\right),
	\end{equation}
where $\lambda_n \geq 0$ is a tuning parameter and $P(\cdot)$ is a penalty function which penalizes ``complexity.''

In our framework, we alter the penalized regression problem slightly. We select a mesh of $m$ knots over the domain of $x$, and use the fitted-values at those knots as our optimization parameters. We replace the penalty function with a finite-difference/Riemann approximation. Fitted values at data points are calculated by cleverly interpolating between fitted values at knots. For brevity, we refer to the general approach as 'meshy optimization.' 
