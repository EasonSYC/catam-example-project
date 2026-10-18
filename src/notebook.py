# %% [markdown]
# # 0.1 Root Finding in One Dimension

# %% [markdown]
# ## Setup

# %%
import matplotlib.pyplot as plt
import matplotlib.collections as mcl
import numpy as np
import pandas as pd

# %%
from collections.abc import Callable
from typing import NamedTuple
import numpy.typing as npt

# %%
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r'\usepackage{amsmath}\usepackage{mathtools}\usepackage{permute}\usepackage[notrig]{physics}\input{macros.tex}',
})

# %%
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.transparent": True,
    "savefig.bbox": 'tight',
})

# %% [markdown]
# ## Question 1

# %%
x: npt.ArrayLike = np.arange(-4.0, -1.0, 0.01)
y1: npt.ArrayLike = 2 * x + 5
y2: npt.ArrayLike = 3 * np.sin(x)

# %%
fig, ax = plt.subplots()

ax.plot(x, y1, label=r'\(y = 2x + 5\)')
ax.plot(x, y2, label=r'\(y = 3 \sin \pqty{x}\)')

ax.set(xlabel=r'\(x\)',
       ylabel=r'\(y\)',
       title=r'Graph of \(3 \sin \pqty{x}\) and \(2x + 5\), for \(-4 \leq x \leq 1\)')
ax.grid()
ax.legend()

fig.savefig('../fig/question-1')

# %% [markdown]
# ## Programming Task: Binary Search

# %%
F = lambda x: 2 * x - 3 * np.sin(x) + 5

# %%
class BinarySearchResult(NamedTuple):
    iteration_count: int # number of iterations
    root_estimate: float # estimate of root
    error: float # truncation error

# %%
def binary_search(
    function: Callable[[float], float], # function for binary search
    left_endpoint: float, # starting left-endpoint
    right_endpoint: float, # starting right-endpoint
    truncation_error: float = 5e-6, # truncation error
    max_iteration: int = 1000 # max iterations
) -> BinarySearchResult:
    iteration: int = 0

    if function(right_endpoint) == 0:
        return BinarySearchResult(0, right_endpoint, 0)
    
    if function(left_endpoint) == 0:
        return BinarySearchResult(0, right_endpoint, 0)

    incr: int

    if function(right_endpoint) > 0 and function(left_endpoint) < 0: # increasing
        incr = 1
    elif function(right_endpoint) < 0 and function(left_endpoint) > 0: # decreasing
        incr = -1
    else:
        raise ValueError
    
    while (np.abs(right_endpoint - left_endpoint) >= 2 * truncation_error) and (iteration < max_iteration):
        iteration += 1

        midpoint: float = (left_endpoint + right_endpoint) / 2

        if function(midpoint) * incr > 0: # on the left
            right_endpoint = midpoint
        elif function(midpoint) * incr < 0: # on the right
            left_endpoint = midpoint
        else: # = 0
            return BinarySearchResult(iteration, midpoint, 0)

    return BinarySearchResult(iteration, (right_endpoint + left_endpoint) / 2, np.abs(right_endpoint - left_endpoint) / 2)

# %%
print(binary_search(F, -3, -2))
print(binary_search(F, -np.pi, 0))
print(binary_search(F, -10, 10))

print(binary_search(F, -4, -1))

# %% [markdown]
# ## What is the Root Itself? [To the best of Float]

# %%
result: BinarySearchResult = binary_search(F, -3, -2, truncation_error=0)
x_ast: float = result.root_estimate
print(result.error, x_ast)

# %% [markdown]
# ## Programming Task: Fixed-Point Iteration

# %%
class IterationProcess(NamedTuple):
    iteration: int # current iteration
    current_root_estimate: float # current value

# %%
def fixed_point_iteration(
    function: Callable[[float], float], # function for fixed point iteration
    x: float, # starting value
    term_difference: float = 1e-5, # end term difference
    max_iteration: int = 1000 # max iterations
) -> list[IterationProcess]:
    iteration: int = 0
    result: list[IterationProcess] = [IterationProcess(iteration, x)]

    while (iteration == 0 or
           np.abs(result[-1].current_root_estimate - result[-2].current_root_estimate) >= term_difference) and (iteration < max_iteration):
        iteration += 1

        x = function(x)
        result.append(IterationProcess(iteration, x))

    return result

# %% [markdown]
# ## Question 3

# %%
fk = lambda x, k: (3 * np.sin(x) + k * x - 5) / (2 + k)
dfk = lambda x, k: (3 * np.cos(x) + k) / (2 + k)

# %%
def fixed_point_iteration_coordinates(root_estimate: pd.DataFrame,
                             function: Callable[[float], float],
                             length: int=-1) -> pd.DataFrame: # convert iteration process to coordinates
    if length == -1:
        length = len(root_estimate.index)
    iteration_process = pd.DataFrame(index=np.arange(0, 2 * length), columns=["x", "y"])

    for i in range(length):
        x: float = root_estimate.current_root_estimate[i]

        if i != 0:
            iteration_process.iloc[2 * i] = (x, x)
            
        iteration_process.iloc[2 * i + 1] = (x, function(x))

    return iteration_process

# %%
def plot_fixed_point_iteration_graph(function,
                         root_estimate: pd.DataFrame,
                         x: npt.ArrayLike,
                         initial_guess: float,
                         root: float,
                         title: str) -> None:
    iteration_process: pd.DataFrame = fixed_point_iteration_coordinates(root_estimate, function)
    y: npt.ArrayLike = function(x)

    ax.plot(x, x, label=r'\(y = x\)')
    ax.plot(x, y, label=r'\(y = f\pqty{x}\)')

    ax.plot(initial_guess, function(initial_guess), marker='+', color='g')
    ax.plot(iteration_process.x,
            iteration_process.y,
            label='Iteration Process',
            marker=',', linewidth=0.5, color='g')

    ax.plot(root, root, marker='+', color='r')

    ax.set(xlabel=r'\(x\)',
        ylabel=r'\(y\)',
        title=title)
    ax.grid()
    ax.legend()

# %%
def truncation_errors(function,
                      root_estimate: pd.DataFrame,
                      root: float) -> pd.DataFrame:
    truncation_errors: pd.DataFrame = pd.DataFrame(root_estimate).drop(columns=["iteration"])

    truncation_errors["function_value"] = function(truncation_errors["current_root_estimate"])
    truncation_errors["truncation_error"] = truncation_errors["current_root_estimate"] - root
    truncation_errors["truncation_ratio"] = truncation_errors["truncation_error"] / truncation_errors["truncation_error"].shift(1)

    
    return truncation_errors

# %% [markdown]
# ### (i): non-convergence

# %%
f = lambda x: fk(x, 0)
df = lambda x: dfk(x, 0)

# %%
root_estimate = pd.DataFrame(fixed_point_iteration(f, -2, 1e-5, 10))
root_estimate

# %%
data: pd.DataFrame = truncation_errors(f, root_estimate, x_ast)
data.to_csv('../data/question-3-i.csv')
data

# %%
fig, ax = plt.subplots()
plot_fixed_point_iteration_graph(f,
                     root_estimate,
                     np.arange(-4.5, -1.0, 0.01),
                     -2,
                     x_ast,
                     r'Iteration with \(k = 0\) and \(x_0 = -2\)')

fig.savefig('../fig/question-3-i')

# %%
df(x_ast)

# %% [markdown]
# ### (iii): Monotonic Convergence

# %%
dfk(x_ast, 3)

# %%
f = lambda x: fk(x, 3)
root_estimate = pd.DataFrame(fixed_point_iteration(f, -2, 1e-5, 20))
root_estimate

# %%
data: pd.DataFrame = truncation_errors(f, root_estimate, x_ast)
data.to_csv('../data/question-3-iii-1.csv')
data

# %%
fig, ax = plt.subplots()
plot_fixed_point_iteration_graph(f,
                     root_estimate,
                     np.arange(-3.5, -1.5, 0.01),
                     -2,
                     x_ast,
                     r'Iteration with \(k = 3\) and \(x_0 = -2\)')

fig.savefig('../fig/question-3-iii-1')

# %% [markdown]
# ### (iii): Oscillatory Convergence

# %%
dfk(x_ast, 1)

# %%
f = lambda x: fk(x, 1)
root_estimate = pd.DataFrame(fixed_point_iteration(f, -2, 1e-5, 20))
root_estimate

# %%
data: pd.DataFrame = truncation_errors(f, root_estimate, x_ast)
data.to_csv('../data/question-3-iii-2.csv')
data

# %%
fig, ax = plt.subplots()
plot_fixed_point_iteration_graph(f,
                     root_estimate,
                     np.arange(-3.5, -1.5, 0.01),
                     -2,
                     x_ast,
                     r'Iteration with \(k = 1\) and \(x_0 = -2\)')

fig.savefig('../fig/question-3-iii-2')

# %% [markdown]
# ### (iv): slow convergence

# %%
f = lambda x: fk(x, 16)
df = lambda x: dfk(x, 16)

root_estimate = pd.DataFrame(fixed_point_iteration(f, -2, 1e-5, 50))
root_estimate

# %%
data: pd.DataFrame = truncation_errors(f, root_estimate, x_ast)
data.to_csv('../data/question-3-iv.csv')
data

# %%
fig, ax = plt.subplots()
plot_fixed_point_iteration_graph(f,
                     root_estimate,
                     np.arange(-3.5, -1.5, 0.01),
                     -2,
                     x_ast,
                     r'Iteration with \(k = 16\) and \(x_0 = -2\)')

fig.savefig('../fig/question-3-iv')

# %%
root_estimate.iloc[-1].current_root_estimate - x_ast

# %%
df(-3), df(-2)

# %%
0.75 * (0.72 ** 34), 1 * (0.82 ** 34)

# %%
dd: float = df(x_ast)
dd

# %%
np.abs(1e-5 * dd / (dd - 1))

# %% [markdown]
# ###  (v): rate of convergence

# %%
dfk(x_ast, 3), dfk(x_ast, 1), dfk(x_ast, 16)

# %% [markdown]
# ## Question 4

# %%
f = lambda x: (- np.pow(x, 3) + 8.5 * np.pow(x, 2) + 8) / 20
df = lambda x: (- 3 * np.pow(x, 2) + 17 * x) / 20
d2f = lambda x: (- 6 * x + 17) / 20

# %%
root_estimate = pd.DataFrame(fixed_point_iteration(f, 5, 1e-10, 5000))
root_estimate

# %%
data: pd.DataFrame = truncation_errors(f, root_estimate, 4)
data['40/(7N e)'] = 40 / (7 * data.index * data.truncation_error)
data.to_csv('../data/question-4.csv')
data

# %%
fig, ax = plt.subplots()
plot_fixed_point_iteration_graph(f,
                     root_estimate,
                     np.arange(3.5, 5, 0.01),
                     5,
                     4,
                     r'Double-Root Iteration with \(x_0 = 5\)')

fig.savefig('../fig/question-4')

# %%
d2: float = d2f(4)
d2

# %%
np.sqrt(2 * (1e-5) / np.abs(d2))

# %% [markdown]
# ## Programming Task: Newton-Raphson

# %%
def newton_raphson_iteration(
    function: Callable[[float], float], # function for newton-raphson iteration
    derivative: Callable[[float], float], # derivative for newton-raphson iteration
    x: float, # starting value
    term_difference: float = 1e-5, # end term difference
    max_iteration: int = 1000 # max iterations
) -> list[IterationProcess]:
    return fixed_point_iteration(lambda x: x - function(x) / derivative(x), x, term_difference=term_difference, max_iteration=max_iteration)

# %% [markdown]
# ## Question 5

# %%
def intermediate_data(function,
                      root_estimate: pd.DataFrame,
                      root: float) -> pd.DataFrame:
    intermediate_data: pd.DataFrame = truncation_errors(function, root_estimate, root)
    intermediate_data["truncation_ratio_squared"] = intermediate_data["truncation_error"] / np.pow(intermediate_data["truncation_error"].shift(1), 2)

    return intermediate_data

# %%
def tangents_verticals(function, derivative, root_estimate: pd.DataFrame, length: int=-1):
    if length == -1:
        length = len(root_estimate.index)
    tangents = []
    verticals = []

    for i in range(length - 1):
        x: float = root_estimate.current_root_estimate[i]
        fx: float = function(x)
        dx: float = derivative(x)

        verticals.append([[x, fx], [x, 0]])
        tangents.append([[x - fx / dx, 0], [x, fx]])

    return (tangents, verticals)

# %%
def plot_newton_raphson_graph(function, derivative, root_estimate: pd.DataFrame,
                              x: npt.ArrayLike, initial_guess: float, root: float,
                              title: str, length: int = -1) -> None:
    tangents, verticals = tangents_verticals(function, derivative, root_estimate, length)

    y: npt.ArrayLike = function(x)

    ax.plot(x, y, label=r'\(y = f\pqty{x}\)')

    ax.plot(initial_guess, function(initial_guess), marker='+')

    ax.add_collection(mcl.LineCollection(tangents, color='g'))
    ax.add_collection(mcl.LineCollection(verticals, color='y', linestyles='dotted'))

    ax.plot(root, 0, marker='+', color='r')

    ax.set(xlabel=r'\(x\)',
           ylabel=r'\(y\)',
           title=title)
    ax.grid()
    ax.legend()

# %% [markdown]
# ### Equation 4

# %%
f = lambda x: 2 * x - 3 * np.sin(x) + 5
df = lambda x: 2 - 3 * np.cos(x)
d2f = lambda x: 3 * np.sin(x)

# %%
root_estimate = pd.DataFrame(newton_raphson_iteration(f, df, -10, max_iteration=100))
root_estimate

# %%
fig, ax = plt.subplots()
plot_newton_raphson_graph(f, df, root_estimate, np.arange(-15, 5, 0.01), -5, x_ast, r'Divergence for \(2x - 3 \sin\pqty{x} + 5\), \(x_0 = -10\)', 6)
fig.savefig('../fig/question-5-eq-4-1-diverge')

# %%
data: pd.DataFrame = intermediate_data(f, root_estimate, x_ast)
data.to_csv('../data/question-5-eq-4-1-diverge.csv')
data

# %%
root_estimate = pd.DataFrame(newton_raphson_iteration(f, df, -2, 1e-20))
root_estimate

# %%
fig, ax = plt.subplots()
plot_newton_raphson_graph(f, df, root_estimate, np.arange(-3.2, -1.9, 0.01), -2, x_ast, r'Convergence for \(2x - 3 \sin\pqty{x} + 5\), \(x_0 = -2\)', 6)
fig.savefig('../fig/question-5-eq-4-2-converge')

# %%
data: pd.DataFrame = intermediate_data(f, root_estimate, x_ast)
data.to_csv('../data/question-5-eq-4-2-converge.csv')
data

# %%
root_estimate = pd.DataFrame(newton_raphson_iteration(f, df, -6))
root_estimate

# %%
data: pd.DataFrame = intermediate_data(f, root_estimate, x_ast)
data.to_csv('../data/question-5-eq-4-3-converge-slow.csv')
data

# %%
d2f(x_ast) / (2 * df(x_ast))

# %% [markdown]
# ### Equation 5

# %%
f = lambda x: np.pow(x, 3) - 8.5 * np.pow(x, 2) + 20 * x - 8
df = lambda x: 3 * np.pow(x, 2) - 17 * x + 20

# %%
root_estimate = pd.DataFrame(newton_raphson_iteration(f, df, 5, 1e-20))
root_estimate

# %%
fig, ax = plt.subplots()
plot_newton_raphson_graph(f, df, root_estimate, np.arange(3.9, 5.1, 0.01), 5, 4, r'Convergence for \(x^3 - 8.5 x^2 + 20 x - 8\), \(x_0 = 5\)', 19)
fig.savefig('../fig/question-5-eq-5')

# %%
data: pd.DataFrame = intermediate_data(f, root_estimate, 4)
data.to_csv('../data/question-5-eq-5.csv')
data

# %% [markdown]
# ## Cleanup

# %%
plt.close('all')


