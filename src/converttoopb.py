from z3 import *
import re
import random
from subprocess import check_output
import os

index = 1
incremental = {}
backwards = {}


def to_pseudo_boolean(expr, toplevel=False):
    if is_const(expr):  # A Boolean variable (e.g., x)
        global incremental
        if expr not in incremental.keys():
            global index
            global backwards
            incremental[expr] = f"x{index}"
            backwards[f"x{index}"] = expr
            index += 1
        if toplevel:
            return f"+1 {incremental[expr]} == 1"
        return f"{incremental[expr]}"
    elif is_eq(expr):
        if toplevel:
            return f"+1 {to_pseudo_boolean(expr.children()[0])} == {to_pseudo_boolean(expr.children()[1])} == 1"
        return f"{to_pseudo_boolean(expr.children()[0])} == {to_pseudo_boolean(expr.children()[1])}"
    elif is_not(expr):  # A negation (e.g., Not(x))
        assert len(expr.children()) == 1
        if toplevel:
            return f"+1 ~{to_pseudo_boolean(expr.children()[0])} == 1"
        return f"~{to_pseudo_boolean(expr.children()[0])}"
    elif is_or(expr):  # An OR clause (e.g., Or(x, y))
        terms = [to_pseudo_boolean(child) for child in expr.children()]
        return "+1 " + " +1 ".join(terms) + " >= 1"
    elif is_and(expr):  # An AND clause (e.g., And(x, y))
        constraints = []
        for child in expr.children():
            constraints.append("+1 " + to_pseudo_boolean(child) + " >= 1")
        return "\n".join(constraints)
    elif (
        is_app_of(expr, Z3_OP_PB_GE)
        or is_app_of(expr, Z3_OP_PB_LE)
        or is_app_of(expr, Z3_OP_PB_EQ)
    ):
        # pdb.set_trace()
        terms = []
        coeffs = expr.params()[1:]  # Get the coefficients and variables
        for i, coeff in enumerate(coeffs):
            constant = coeff
            if is_app_of(expr, Z3_OP_PB_LE):
                constant = -constant
            variable = to_pseudo_boolean(expr.arg(i))  # Get the variable
            suffix = ""
            if constant > 0:
                suffix = "+"
            terms.append(f"{suffix}{constant} {variable}")
        rhs = expr.params()[0]  # The right-hand side of the >= constraint

        if is_app_of(expr, Z3_OP_PB_LE):
            rhs = -rhs
        rel_op = ">="
        if is_app_of(expr, Z3_OP_PB_EQ):
            rel_op = "=="
        return " ".join(terms) + f" {rel_op} {rhs}"
    elif is_app_of(expr, Z3_OP_PB_AT_MOST):
        terms = []

        for i in range(expr.num_args()):
            variable = to_pseudo_boolean(expr.arg(i))  # Get the variable
            terms.append(f"+1 {variable}")
        rhs = expr.params()[0]  # The right-hand side of the >= constraint
        rel_op = "<="
        return " ".join(terms) + f" {rel_op} {rhs}"
    else:
        pdb.set_trace()
        raise NotImplementedError(f"Unhandled expression type: {expr}")


import pdb


def convert(cons):
    pb_constraints = []
    t = Tactic("tseitin-cnf")
    for constraint in cons:
        try:
            cnfs = t(constraint)[0]
            for cnf in cnfs:
                pb_constraints.append(to_pseudo_boolean(cnf, True))
        except Exception as e:
            pdb.set_trace()
            print(e)
    header = f"* #variable= {index} #condition= {len(pb_constraints)}\n"
    return header + ";\n".join(pb_constraints) + ";"


class RoundingModel:
    def __init__(self):
        self.sat = False
        self.assign = {}

    def __bool__(self):
        return self.sat

    def __getitem__(self, i):
        return self.assign[i]


def roundingSolve(cons, soplex=False):
    filename = f"roundingtemp_{random.randint(0, 1000)}.opb"
    with open(filename, "w") as f:
        print(convert(cons), file=f)
    args = ["./roundingsat", filename, "--print-sol=1", "--verbosity=0"]
    if soplex:
        args[0] = "./roundingsatsoplex"
    roundingout = check_output(args)
    os.remove(filename)
    resModel = RoundingModel()
    roundingout = str(roundingout, encoding="utf-8")

    if re.search(r"UNSATISFIABLE", roundingout):
        resModel.sat = False
        return resModel
    else:
        resModel.sat = True

    solution = roundingout.split()[3:]
    for i in solution:
        if i[0] == "-":
            resModel.assign[backwards[i[1:]]] = False
        else:
            resModel.assign[backwards[i]] = True
    return resModel
