# coding=utf8


class DataType:
    INT = "int"
    INT_LIST = "int[]"
    BOOL = "bool"
    FN_LIST_TO_INT = "int[]->int"
    FN_INT_TO_INT = "int->int"
    FN_INT_TO_BOOL = "int->bool"
    FN_INT_INT_TO_INT = "int->int->int"


class FuncDefinition:
    def __init__(self, func, arguments, return_type):
        """
        :param func:        Callable function
        :param arguments:   Function Arguments
                            [DataType, DataType, ..]
        :param return_type: DataType
        """
        self.func = func
        self.arguments = arguments
        self.arguments_num = len(arguments)
        self.return_type = return_type


def scan(f, v):
    if len(v):
        y = [v[0]] * len(v)
        for i in range(1, len(v)):
            y[i] = f(y[i-1], v[i])
        return y
    return None


Functions = {
    "FO": {
        "HEAD": {
            "func": lambda v: v[0] if len(v) > 0 else None,
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT
        },
        "LAST": {
            "func": lambda v: v[-1] if len(v) > 0 else None,
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT
        },
        "TAKE": {
            "func": lambda n, v: v[:n],
            "arguments": [DataType.INT, DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "DROP": {
            "func": lambda n, v: v[n:],
            "arguments": [DataType.INT, DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "ACCESS": {
            "func": lambda n, v: v[n] if n >= 0 and len(v) > 0 else None,
            "arguments": [DataType.INT, DataType.INT_LIST],
            "return_type": DataType.INT
        },
        "MINIMUM": {
            "func": lambda v: min(v) if len(v) > 0 else None,
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT
        },
        "MAXIMUM": {
            "func": lambda v: max(v) if len(v) > 0 else None,
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT
        },
        "REVERSE": {
            "func": lambda v: list(reversed(v)),
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "SORT": {
            "func": lambda v: sorted(v),
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "SUM": {
            "func": lambda v: sum(v),
            "arguments": [DataType.INT_LIST],
            "return_type": DataType.INT
        }
    },
    "HO": {
        "MAP": {
            "func": lambda f, v: [f(x) for x in v],
            "arguments": [DataType.FN_INT_TO_INT, DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "FILTER": {
            "func": lambda f, v: [x for x in v if f(x)],
            "arguments": [DataType.FN_INT_TO_BOOL, DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "COUNT": {
            "func": lambda f, v: len([x for x in v if f(x)]),
            "arguments": [DataType.FN_INT_TO_BOOL, DataType.INT_LIST],
            "return_type": DataType.INT
        },
        "ZIPWITH": {
            "func": lambda f, v1, v2: [f(x, y) for x, y in zip(v1, v2)],
            "arguments": [DataType.FN_INT_INT_TO_INT, DataType.INT_LIST, DataType.INT_LIST],
            "return_type": DataType.INT_LIST
        },
        "SCAN1": {
            "func": scan,
            "arguments": [DataType.FN_LIST_TO_INT],
            "return_type": DataType.INT_LIST
        }
    },
    "lambdas": {
        "(+1)": {
            "func": lambda x: x + 1,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(-1)": {
            "func": lambda x: x - 1,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(*2)": {
            "func": lambda x: x * 2,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(/2)": {
            "func": lambda x: x / 2,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(*(-1))": {
            "func": lambda x: x * (-1),
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(**2)": {
            "func": lambda x: x**2,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(*3)": {
            "func": lambda x: x * 3,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(/3)": {
            "func": lambda x: x / 3,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(*4)": {
            "func": lambda x: x * 4,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(/4)": {
            "func": lambda x: x / 4,
            "arguments": [DataType.INT],
            "return_type": DataType.INT
        },
        "(>0)": {
            "func": lambda x: x > 0,
            "arguments": [DataType.INT],
            "return_type": DataType.BOOL
        },
        "(<0)": {
            "func": lambda x: x < 0,
            "arguments": [DataType.INT],
            "return_type": DataType.BOOL
        },
        "(%2==0)": {
            "func": lambda x: x % 2 == 0,
            "arguments": [DataType.INT],
            "return_type": DataType.BOOL
        },
        "(%2==1)": {
            "func": lambda x: x % 2 == 1,
            "arguments": [DataType.INT],
            "return_type": DataType.BOOL
        },
        "(+)": {
            "func": lambda x, y: x + y,
            "arguments": [DataType.INT, DataType.INT],
            "return_type": DataType.INT
        },
        "(-)": {
            "func": lambda x, y: x - y,
            "arguments": [DataType.INT, DataType.INT],
            "return_type": DataType.INT
        },
        "(*)": {
            "func": lambda x, y: x * y,
            "arguments": [DataType.INT, DataType.INT],
            "return_type": DataType.INT
        },
        "MIN": {
            "func": lambda x, y: min(x, y),
            "arguments": [DataType.INT, DataType.INT],
            "return_type": DataType.INT
        },
        "MAX": {
            "func": lambda x, y: max(x, y),
            "arguments": [DataType.INT, DataType.INT],
            "return_type": DataType.INT
        }
    }
}
