import ir
ir.generate_asdl_file()

import _asdl.loma as loma_ir
import autodiff
import pretty_print

from reverse_diff import reverse_diff

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler
import ctypes
import error
import math
import slang_utils
import slangpy
import unittest
import numpy as np

def count_atomic_adds(func_str: str) -> int:
    return func_str.count("atomic_add")


def main():
    # Simple primal function:
    #
    # def square(x : In[float]) -> float:
    #     return x * x
    #
    # This is a good first test because reverse-mode AD should generate
    # two gradient contributions into x.
    x_arg = loma_ir.Arg("x", loma_ir.Float(), loma_ir.In())

    square_func = loma_ir.FunctionDef(
        "square",
        [x_arg],
        [
            loma_ir.Return(
                loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    loma_ir.Var("x", t=loma_ir.Float()),
                    loma_ir.Var("x", t=loma_ir.Float()),
                    t=loma_ir.Float(),
                )
            )
        ],
        is_simd=False,
        ret_type=loma_ir.Float(),
    )

    structs = {}
    funcs = {
        "square": square_func,
    }
    diff_structs = {}
    func_to_rev = {
        "square": "d_square",
    }

    d_square = reverse_diff(
        "d_square",
        structs,
        funcs,
        diff_structs,
        square_func,
        func_to_rev,
    )

    print("===== GENERATED REVERSE FUNCTION =====")

    # Depending on your pretty_print.py, one of these may be the correct call.
    # Try the first one. If it errors, comment it out and try the second.
    generated = pretty_print.loma_to_str(d_square)
    # generated = pretty_print.pretty_print(d_square)

    print(generated)
    print("===== SUMMARY =====")
    print("atomic_add count:", count_atomic_adds(generated))


if __name__ == "__main__":
    main()