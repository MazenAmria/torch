from typing import Union
from numbers import Number

from pygraphviz import AGraph

from autograd import Node, Variable
from autograd.operations import Addition, Multiplication, Negation, Exponentiation, NaturalLogarithm


def add_binary_operation(node: Node, graph: AGraph, label: str, ref: str) -> AGraph:
    graph.add_node(ref, label=label)
    graph = visualize_node(node.a, graph, ref + '1')
    graph = visualize_node(node.b, graph, ref + '2')
    graph.add_edge(ref + '1', ref)
    graph.add_edge(ref + '2', ref)
    return graph


def add_unary_operation(node: Node, graph: AGraph, label: str, ref: str) -> AGraph:
    graph.add_node(ref, label=label)
    graph = visualize_node(node.x, graph, ref + '1')
    graph.add_edge(ref + '1', ref)
    return graph


def visualize_node(node: Union[Node, Number], graph: AGraph, ref: str) -> AGraph:
    if isinstance(node, Number):
        graph.add_node(ref, label=f'{node:.3f}')
    elif isinstance(node, Variable):
        if node.name is not None:
            graph.add_node(ref, label=f'{{ <f0> {node.name} }} | {{ <f1> v | <f2> g }} | {{ <f3> {node.value:.3f} | <f4> {node.grad:.3f} }}', shape='record')
        else:
            graph.add_node(ref, label=f'{{ <f0> v | <f1> g }} | {{ <f2> {node.value:.3f} | <f3> {node.grad:.3f} }}', shape='record')
        if node.parent is not None:
            graph = visualize_node(node.parent, graph, ref + '1')
            graph.add_edge(ref + '1', ref)
    elif isinstance(node, Addition):
        graph = add_binary_operation(node, graph, '+', ref)
    elif isinstance(node, Multiplication):
        graph = add_binary_operation(node, graph, '×', ref)
    elif isinstance(node, Exponentiation):
        graph = add_binary_operation(node, graph, '^', ref)
    elif isinstance(node, Negation):
        graph = add_unary_operation(node, graph, '× -1', ref)
    elif isinstance(node, NaturalLogarithm):
        graph = add_unary_operation(node, graph, 'ln', ref)
    else:
        raise TypeError(f'Cannot Visualize node of type {type(node)}')

    return graph


def visualize(variable: Variable, path: str) -> None:
    graph = AGraph(directed=True)

    graph = visualize_node(variable, graph, 'n')

    graph.layout(prog='dot')
    graph.draw(path)
