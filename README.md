# pearcey
A python package to compute the Pearcey function/integral in catastrophe optics

The Pearcey function $\text{Pe}(x, y)$ is defined as

$$\text{Pe}(x, y) = \int_{-\infty}^{\infty}e^{i(t^4 + xt^2 + yt)} dt$$

![pearcey](https://github.com/user-attachments/assets/bcefda01-fd9e-4323-95a2-b9e34a3b88e6)

## Requirements
- numpy
- scipy (preferrably >= 1.10.0 for a better numerical integration scheme)

## Installation
This package is available on PyPI. To install, simply run
```
pip install pearcey
```

## Usage
The core function is `pearcey(x, y)`. For example, to compute the value of the Pearcey function $\text{Pe}(x, y)$ at $x = 0$ and $y = 0$, simply do
```python
>>> from pearcey import pearcey; pearcey(0, 0)
(1.6748133935381728+0.693730422047619j)
```

For details, refer to the docstring [here](https://github.com/ricokaloklo/pearcey/blob/main/pearcey/pearcey.py#L142).
