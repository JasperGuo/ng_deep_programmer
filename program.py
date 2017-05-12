# coding=utf8

from program_generator.build_graph import Graph


class TestCase:
    def __init__(self, inputs, output):
        """
        :param inputs:
            [
                {
                    "variable_name": "",
                    "data_type": "",
                    "value":
                }
            ]
        :param output:
            {
                "variable_name": "",
                "data_type": "",
                "value":
            }
        """
        self.inputs = inputs
        self.output = output

    def serialize(self):
        return {
            "inputs": self.inputs,
            "output": self.output,
        }

    @classmethod
    def deserialize(cls, test_case_json):
        return cls(**test_case_json)


class Program:
    def __init__(self, graph, tree, topological_sort_result):
        self.graph = graph
        self.tree = tree
        self.topological_sort_result = topological_sort_result
        self.test_cases = list()

    def print(self):
        for v in self.topological_sort_result:
            func = self.graph.get_attr(v, "func")
            variable_name = self.graph.get_attr(v, "variable_name")
            if not func:
                # Input node
                print(variable_name, " = ", self.graph.get_attr(v, "data_type"))
            else:

                lambda_exp = self.graph.get_attr(v, "lambda")

                if not lambda_exp:
                    print(variable_name, " = ", func, '(',
                          ', '.join([self.graph.get_attr(v, "variable_name") for v in self.graph.parents(v)]), ")")
                else:
                    print(variable_name, " = ", func, '(', lambda_exp, ",",
                          ', '.join([self.graph.get_attr(v, "variable_name") for v in self.graph.parents(v)]), ")")

    def to_string(self):
        string = list()
        for v in self.topological_sort_result:
            func = self.graph.get_attr(v, "func")
            variable_name = self.graph.get_attr(v, "variable_name")
            if not func:
                # Input node
                string.append(','.join([variable_name, self.graph.get_attr(v, "data_type")]))
            else:

                lambda_exp = self.graph.get_attr(v, "lambda")

                if not lambda_exp:
                    string.append(','.join([variable_name, func, ','.join(
                        [self.graph.get_attr(v, "variable_name") for v in self.graph.parents(v)])]))
                else:
                    string.append(','.join([variable_name, func, lambda_exp, ','.join(
                        [self.graph.get_attr(v, "variable_name") for v in self.graph.parents(v)])]))
        return '\n'.join(string)

    def inputs(self):
        _inputs = list()
        for vertex in self.graph.vertices:
            func = self.graph.get_attr(vertex, "func")
            if not func:
                _inputs.append({
                    "vertex": vertex,
                    "data_type": self.graph.get_attr(vertex, "data_type"),
                    "variable_name": self.graph.get_attr(vertex, "variable_name")
                })
        return _inputs

    def output(self):

        for vertex in self.graph.vertices:
            out_degree = self.graph.outdegree(vertex)
            if out_degree == 0:
                return {
                    "vertex": vertex,
                    "data_type": self.graph.get_attr(vertex, "data_type"),
                    "variable_name": self.graph.get_attr(vertex, "variable_name")
                }

    def expressions(self):
        """
        Get program expression
        :return:
        """
        exprs = list()
        for v in self.topological_sort_result:
            func = self.graph.get_attr(v, "func")
            variable_name = self.graph.get_attr(v, "variable_name")
            if func:

                lambda_exp = self.graph.get_attr(v, "lambda")

                if not lambda_exp:
                    expr = [variable_name, func] + [self.graph.get_attr(v, "variable_name") for v in
                                                    self.graph.parents(v)]
                else:
                    expr = [variable_name, func, lambda_exp] + [self.graph.get_attr(v, "variable_name") for v in
                                                                self.graph.parents(v)]
                exprs.append(expr)
        return exprs

    def add_test_case(self, test_case):
        self.test_cases.append(test_case)

    def funcs(self):
        func_set = set()
        for vertex in self.graph.vertices:
            func = self.graph.get_attr(vertex, "func")
            if func:
                func_set.add(func)
                lambda_exp = self.graph.get_attr(vertex, "lambda")
                func_set.add(lambda_exp)
        return func_set

    @property
    def length(self):
        return len(self.expressions())

    def serialize(self):
        """
        Serialize Program Instance to json
        :return:
        """
        return {
            "graph": self.graph.serialize(),
            "tree": self.tree,
            "topological_sort_result": self.topological_sort_result,
            "test_cases": [case.serialize() for case in self.test_cases]
        }

    @classmethod
    def deserialize(cls, program_json):
        graph = Graph.deserialize(program_json["graph"])
        program = cls(graph, program_json["tree"], program_json["topological_sort_result"])
        for case in program_json["test_cases"]:
            program.add_test_case(TestCase.deserialize(case))
        return program
