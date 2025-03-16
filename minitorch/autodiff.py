from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

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
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")
    arg1 = [i for i in vals]
    arg1[arg] += epsilon
    m = f(*arg1)
    arg1[arg] -= 2 * epsilon
    n = f(*arg1)
    return (m - n) / (2 * epsilon)
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
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    order = []
    seen = set()

    queue = deque()
    queue.append(variable)
    while queue:
        node = queue.popleft()
        if node.is_constant() or seen.__contains__(node.unique_id):
            continue
        order.append(node)
        seen.add(node.unique_id)
        if not node.is_leaf():
            for parent in node.parents:
                queue.append(parent)

    return order

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    order = topological_sort(variable)
    current_variable_deriv = {variable.unique_id : deriv}

    for o in order:
        if o.unique_id not in current_variable_deriv:
            # 叶子节点
            continue
        d = current_variable_deriv[o.unique_id]
        parent_derivs = o.chain_rule(d)
        for parent, der in parent_derivs:
            if parent.is_leaf():
                parent.accumulate_derivative(der)
            else:
                if current_variable_deriv.__contains__(parent.unique_id):
                    current_variable_deriv[parent.unique_id] += der
                else:
                    current_variable_deriv[parent.unique_id] = der


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
