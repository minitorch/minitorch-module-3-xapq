from dataclasses import dataclass
from typing import Any, Iterable, Tuple
from collections import defaultdict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    term1 = f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1:])
    term2 = f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1:])
    return (term1 - term2) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    ordered_vars = []
    sorted_ids = set()

    def topsort(var: Variable):
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant() and parent.unique_id not in sorted_ids:
                    topsort(parent)
        sorted_ids.add(var.unique_id)
        ordered_vars.append(var)

    topsort(variable)
    return ordered_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives = defaultdict(float)
    derivatives[variable.unique_id] = deriv
    ordered_vars = topological_sort(variable)
    for var in ordered_vars[::-1]:
        var_derivative = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(var_derivative)
        else:
            for parent_var, parent_d in var.chain_rule(var_derivative):
                derivatives[parent_var.unique_id] += parent_d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
