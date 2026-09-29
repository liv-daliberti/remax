# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

import ast
from dataclasses import dataclass
import functools
import json
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
import queue
import re
import signal
import threading
from collections import Counter
from fractions import Fraction
from itertools import islice, zip_longest
from math import isclose
from typing import Any, Optional

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from math_verify import grader as math_verify_grader
from math_verify import parser as math_verify_parser
from math_verify.errors import TimeoutException as MathVerifyTimeout
from math_verify.utils import timeout as math_verify_signal_timeout
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from .math_route import validate_math_route_response
from .benchmark import (
    EvaluationFailure,
    failure,
    grade_reference_response,
    outcome_key,
    parse_reference,
    reward_from_diagnostic,
    validate_reward_batch,
)

# Public legacy text extractors are used only for ordinary MATH formatting.
# Executable benchmark grading always goes through modebench.api in benchmark.py.
from modebench.grading import extract_answer, VerifiedExplorationIdentity

validated_modebench_outcome_key = outcome_key

# math_verify catches its own TimeoutException and returns False/[] internally.
# Remember deadlines per calling thread so the outer reward boundary can refuse
# that numeric fallback even when the library suppresses the exception.
_math_verify_deadline = threading.local()


def _thread_compatible_math_verify_timeout(timeout_seconds: int = 10):
    """Keep math_verify bounded when grading runs in an actor worker thread.

    math_verify's POSIX timeout installs SIGALRM. Python only permits signal
    handler installation on the interpreter's main thread, while OAT grades
    actor responses in a ThreadPool to avoid forking a CUDA/vLLM process. On
    the main thread retain math_verify's native alarm. Else run the protected
    operation in a daemon helper thread and bound the caller's wait.
    """

    seconds = int(timeout_seconds)
    if seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    def decorator(func):
        signal_wrapped = math_verify_signal_timeout(timeout_seconds=seconds)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if threading.current_thread() is threading.main_thread():
                try:
                    return signal_wrapped(*args, **kwargs)
                except MathVerifyTimeout:
                    _math_verify_deadline.expired = True
                    raise
            result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

            def run() -> None:
                try:
                    result_queue.put((True, func(*args, **kwargs)))
                except BaseException as error:
                    result_queue.put((False, error))

            worker = threading.Thread(
                target=run,
                name="math-verify-timeout",
                daemon=True,
            )
            worker.start()
            worker.join(seconds)
            if worker.is_alive():
                _math_verify_deadline.expired = True
                raise MathVerifyTimeout("Operation timed out!")
            succeeded, value = result_queue.get_nowait()
            if succeeded:
                return value
            if isinstance(value, MathVerifyTimeout):
                _math_verify_deadline.expired = True
            raise value

        return wrapper

    return decorator


# parse() and verify() resolve `timeout` from their defining module globals at
# call time. Install the actor-thread-safe policy without altering the external
# math_verify package or its grading semantics.
math_verify_parser.timeout = _thread_compatible_math_verify_timeout
math_verify_grader.timeout = _thread_compatible_math_verify_timeout


_math_verify_sympy_solve_and_compare = math_verify_grader.sympy_solve_and_compare


def _robust_math_verify_sympy_solve_and_compare(
    gold,
    pred,
    float_rounding: int,
    numeric_precision: int,
):
    """Repair math_verify's scalar ``solve(Eq)`` compatibility edge case.

    Some SymPy versions return a list of scalar roots for ``solve(Eq, symbols)``
    while math_verify assumes a list of dictionaries and calls ``.items()``.
    Retry only that failing equality path with ``dict=True`` and preserve the
    package's exact symbol-key and expression-comparison semantics.
    """

    try:
        return _math_verify_sympy_solve_and_compare(
            gold,
            pred,
            float_rounding,
            numeric_precision,
        )
    except AttributeError as error:
        if (
            "items" not in str(error)
            or not isinstance(gold, sympy.Eq)
            or not isinstance(pred, sympy.Eq)
        ):
            raise

    solved_gold = sympy.solve(
        gold,
        gold.free_symbols,
        dict=True,
    )
    solved_pred = sympy.solve(
        pred,
        pred.free_symbols,
        dict=True,
    )
    if not isinstance(solved_gold, list) or not isinstance(solved_pred, list):
        return False
    if len(solved_gold) != len(solved_pred):
        return False

    unmatched = list(solved_pred)
    for gold_solution in solved_gold:
        if not isinstance(gold_solution, dict):
            return False
        match_index = None
        for index, pred_solution in enumerate(unmatched):
            if (
                isinstance(pred_solution, dict)
                and set(gold_solution) == set(pred_solution)
                and all(
                    math_verify_grader.sympy_expr_eq(
                        gold_solution[key],
                        pred_solution[key],
                        float_rounding,
                        numeric_precision,
                    )
                    for key in gold_solution
                )
            ):
                match_index = index
                break
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


math_verify_grader.sympy_solve_and_compare = _robust_math_verify_sympy_solve_and_compare


def collect_threaded_math_rewards(
    pool,
    reward_fn,
    responses,
    references,
    *,
    timeout_seconds: float,
):
    """Grade a response batch concurrently with one bounded wait per result."""

    responses, references = list(responses), list(references)
    if len(responses) != len(references):
        raise failure("invalid_reference", "response/reference batch lengths differ")
    # ModeBench owns its bounded subprocess. Do not race its cold start with
    # the historical one-second ThreadPool deadline for ordinary MATH.
    pending = []
    for index, (response, reference) in enumerate(zip(responses, references)):
        try:
            pending.append(
                reward_fn(response, reference)
                if parse_reference(reference) is not None
                else pool.apply_async(reward_fn, (response, reference))
            )
        except EvaluationFailure as error:
            raise EvaluationFailure(
                dict(error.diagnostic, context="reward_submission", row_index=index)
            ) from error
        except Exception as error:
            raise failure(
                "worker_failure", f"{type(error).__name__}: {error}", row_index=index
            ) from error
    rewards, infos = [], []
    for index, result in enumerate(pending):
        try:
            info, reward = (
                result
                if isinstance(result, tuple)
                else result.get(timeout=timeout_seconds)
            )
        except EvaluationFailure as error:
            raise EvaluationFailure(
                dict(error.diagnostic, context="reward_batch", row_index=index)
            ) from error
        except (MultiprocessingTimeoutError, TimeoutError, MathVerifyTimeout) as error:
            raise failure(
                "timeout", "reward worker deadline exceeded", row_index=index
            ) from error
        except Exception as error:
            raise failure(
                "worker_failure", f"{type(error).__name__}: {error}", row_index=index
            ) from error
        rewards.append(reward)
        infos.append(info)
    validate_reward_batch(
        rewards,
        infos,
        count=len(responses),
        context="threaded_rewards",
        references=references,
    )
    return rewards, infos


# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            # use regex, the prefix should be either the start of the string or a non-alphanumeric character
            # the suffix should be either the end of the string or a non-alphanumeric character
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def repeatness(s: str):
    def ranks(values):
        index = {v: i for i, v in enumerate(sorted(set(values)))}
        return [index[v] for v in values]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        self._armed = False
        self._old_handler = None

    def handle_timeout(self, signum, frame):
        _math_verify_deadline.expired = True
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        self._old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        self._armed = True
        return self

    def __exit__(self, type, value, traceback):
        if self._armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)
            self._armed = False


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def _is_latex_equal(str1, str2):
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:  # noqa
            return norm1 == norm2
    return False


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    _math_verify_deadline.expired = False
    try:
        with timeout(1):
            try:
                if (len(given_answer) > 128 and repeatness(given_answer)) or (
                    len(ground_truth) > 128 and repeatness(ground_truth)
                ):
                    return False
                # First conduct normalized string matching.
                ground_truth_normalized = _normalize(ground_truth)
                given_normalized = _normalize(given_answer)
                if ground_truth_normalized is None:
                    return False
                if ground_truth_normalized == given_normalized:
                    return True

                # Next call math verify.
                given_answer.replace("\n", "")
                ground_truth.replace("\n", "")
                if "$" not in given_answer:
                    given_answer = f"${given_answer}$"
                if "$" not in ground_truth:
                    ground_truth = f"${ground_truth}$"
                return verify(
                    parse(
                        ground_truth,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    parse(
                        given_answer,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    timeout_seconds=1,
                )
                # or symbolic_equal(ground_truth, given_answer)
            except (TimeoutError, MathVerifyTimeout, EvaluationFailure):
                raise
            except Exception:
                return False
    except (TimeoutError, MathVerifyTimeout) as error:
        raise failure("timeout", "full MATH verifier deadline exceeded") from error
    finally:
        if getattr(_math_verify_deadline, "expired", False):
            _math_verify_deadline.expired = False
            raise failure("timeout", "math_verify suppressed an internal deadline")


def is_value_equal(given_answer: str, ground_truth: str) -> bool:
    assert ground_truth is not None
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    str_equal = ground_truth_normalized_mathd == given_answer_normalized_mathd
    try:
        number_equal = float(ground_truth_normalized_mathd) == float(
            given_answer_normalized_mathd
        )
        return str_equal or number_equal
    except Exception:
        return str_equal


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def _clean_final_answer_candidate(candidate: str | None) -> str | None:
    """Return a compact final-answer candidate, or ``None`` for malformed text."""

    if candidate is None:
        return None
    candidate = str(candidate).strip()
    if not candidate:
        return None
    # Answer identity is deliberately stricter than correctness grading: mode
    # metrics need one stable, compact key per formatted final answer.
    if "\n" in candidate or "\r" in candidate or "<" in candidate or ">" in candidate:
        return None
    if len(candidate) > 160:
        return None
    candidate = re.sub(
        r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    return candidate or None


def _extract_r1_reasoning_and_answer_sections(
    model_response: str,
) -> tuple[str | None, str | None]:
    """Return ``(<think> body, <answer> body)`` for an R1-style trace.

    Training prompts already end inside an open ``<think>`` tag, so many model
    responses begin with reasoning text and only emit the closing ``</think>``
    before ``<answer>``. We accept both:

    1. full tagged traces containing ``<think>...</think><answer>...</answer>``
    2. response-only continuations containing ``... </think> <answer>...</answer>``
    """

    if "<answer>" not in model_response or "</answer>" not in model_response:
        return None, None
    try:
        prefix, answer_suffix = model_response.split("<answer>", 1)
        answer = answer_suffix.split("</answer>", 1)[0].strip()
        reasoning = None
        if "<think>" in prefix and "</think>" in prefix:
            reasoning = prefix.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        elif "</think>" in prefix:
            reasoning = prefix.split("</think>", 1)[0].strip()
        else:
            return None, None
        answer = answer
    except Exception:
        return None, None
    return reasoning or None, answer or None


def extract_normalized_final_answer(
    model_response: str, *, template: str = "r1", gt_answer: Any = None
) -> str | None:
    """Return a conservative canonical key for an answer or exact benchmark mode."""

    # Keep the benchmark call outside ordinary-MATH best-effort parsing.
    # Infrastructure failure must propagate, never become a missing identity.
    if parse_reference(gt_answer) is not None:
        return outcome_key(model_response, gt_answer)
    try:
        candidate = None
        if template == "r1":
            # Match the training reward's strict R1 formatting gate.
            _, candidate = _extract_r1_reasoning_and_answer_sections(model_response)
            if candidate is None:
                return None
        else:
            candidate = extract_answer(model_response)
        if candidate is None:
            return None
        extracted = extract_answer(candidate) if "\\boxed" in candidate else candidate
        extracted = _clean_final_answer_candidate(extracted)
        if extracted is None:
            return None
        normalized = normalize_final_answer(extracted)
        normalized = _normalize(normalized)
        if normalized is None:
            normalized = normalize_final_answer(extracted)
        if normalized is None:
            return None
        normalized = str(normalized).strip().lower()
        return normalized or None
    except Exception:
        return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(
        model_answer, gt_answer
    )
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct


def boxed_reward_fn(model_response, gt_answer, fast=False):
    diagnostic = grade_reference_response(model_response, gt_answer)
    if diagnostic is not None:
        return reward_from_diagnostic(diagnostic)
    model_answer = extract_answer(model_response)
    if model_answer is None:
        return {"formatted": False}, 0.0
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {"formatted": True}, 1.0  # Correctness reward.
    else:
        return {
            "formatted": True
        }, 0.0  # Formatted but wrong answer; no format reward to avoid hacking.


def validated_math_route_signature(
    model_response: str,
    problem: str,
    gt_answer: Any,
    *,
    fast: bool = False,
    task_verified: bool | None = None,
) -> str | None:
    """Return an executable route identity only for a task-correct response.

    Task correctness, route execution, and agreement between the trace terminal
    value and the response's boxed answer are independent fail-closed checks.
    A correct answer without a valid trace keeps its ordinary task reward but
    has no route identity.
    """

    try:
        if task_verified is None:
            _info, reward = boxed_reward_fn(
                model_response,
                gt_answer,
                fast=fast,
            )
            task_verified = float(reward) > 0.0
        if not bool(task_verified):
            return None
        validation = validate_math_route_response(model_response, problem)
        if validation is None:
            return None
        model_answer = extract_answer(model_response)
        if model_answer is None:
            return None
        terminal_answer = sympy.latex(validation.terminal_value)
        if not grade(model_answer, terminal_answer, fast=fast):
            return None
        return validation.route_signature
    except EvaluationFailure:
        raise
    except Exception:
        return None


def validated_exploration_identity(
    model_response: str,
    problem: str,
    gt_answer: Any,
    *,
    fast: bool = False,
    task_verified: bool | None = None,
) -> VerifiedExplorationIdentity | None:
    """Return the endpoint/route pair for ModeBench or free-form MATH.

    ModeBench references always remain on their exact executable validator
    path. Free-form MATH may reuse the actor's already-computed task verdict,
    but route admission still independently executes the restricted trace and
    checks its terminal value against the boxed response. Thus a task-correct
    response without a valid route retains an endpoint identity and ordinary
    reward while contributing no route identity.
    """

    if parse_reference(gt_answer) is not None:
        # Historical route signatures are not part of the maintained API.
        # Gate them with structured grading first; quarantine the legacy helper.
        from .legacy_modebench import exploration_identity

        return exploration_identity(model_response, gt_answer)
    try:
        if task_verified is None:
            _info, reward = boxed_reward_fn(
                model_response,
                gt_answer,
                fast=fast,
            )
            task_verified = float(reward) > 0.0
        if not bool(task_verified):
            return None
        endpoint = extract_normalized_final_answer(
            model_response,
            template="qwen_math_route",
            gt_answer=gt_answer,
        )
        if endpoint is None:
            return None
        route = validated_math_route_signature(
            model_response,
            problem,
            gt_answer,
            fast=fast,
            task_verified=True,
        )
        return VerifiedExplorationIdentity(
            verifier="math_verify",
            endpoint_key=f"math-answer:{endpoint}",
            route_signature=route,
        )
    except EvaluationFailure:
        raise
    except Exception:
        return None


def _tag_format_rejection(diagnostic, *, formatted=False):
    # The executable verdict was obtained first so a failed environment cannot
    # hide behind an early formatting rejection. Then apply the legacy gate.
    if diagnostic is None:
        return {'formatted': formatted}, 0.0
    info, reward = reward_from_diagnostic(dict(
        diagnostic, status='malformed', verified=False, canonical_key=None,
        detail='response violates the historical answer-tag format contract'))
    info['formatted'] = formatted
    return info, reward


def answer_tag_reward_fn(model_response, gt_answer, fast=False):
    benchmark_result = grade_reference_response(model_response, gt_answer)
    # We are strict about format to evaluate our models.
    if "</think> <answer>" in model_response and "</answer>" in model_response:
        model_answer = model_response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in model_answer:
            model_answer = extract_answer(model_answer)
            if model_answer is None:
                return _tag_format_rejection(benchmark_result, formatted=True)
        if benchmark_result is not None:
            # Tags are a historical formatting contract; grade only their body.
            return reward_from_diagnostic(
                grade_reference_response(model_answer, gt_answer)
            )
        if isinstance(gt_answer, float) or isinstance(gt_answer, int):
            gt_answer = str(gt_answer)
        if isinstance(gt_answer, str):
            is_correct = grade(model_answer, gt_answer, fast)
        elif isinstance(gt_answer, list):
            is_correct = False
            for gt in gt_answer:
                is_correct |= grade(model_answer, gt, fast)
        if is_correct:
            return {"formatted": True}, 1.0  # Correctness reward.
        else:
            return (
                {"formatted": True},
                0.0,
            )  # Formatted but wrong answer; no format reward to avoid hacking.
    else:
        return _tag_format_rejection(benchmark_result)


def answer_tag_reward_fn_for_orz(model_response, gt_answer, fast=False):
    benchmark_result = grade_reference_response(model_response, gt_answer)
    # We are a bit less strict for baselines.
    if "<answer>" in model_response and "</answer>" in model_response:
        model_answer = model_response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in model_answer:
            model_answer = extract_answer(model_answer)
            if model_answer is None:
                return _tag_format_rejection(benchmark_result, formatted=True)
        if benchmark_result is not None:
            # Tags are a historical formatting contract; grade only their body.
            return reward_from_diagnostic(
                grade_reference_response(model_answer, gt_answer)
            )
        if isinstance(gt_answer, float) or isinstance(gt_answer, int):
            gt_answer = str(gt_answer)
        if isinstance(gt_answer, str):
            is_correct = grade(model_answer, gt_answer, fast)
        elif isinstance(gt_answer, list):
            is_correct = False
            for gt in gt_answer:
                is_correct |= grade(model_answer, gt, fast)
        if is_correct:
            return {"formatted": True}, 1.0  # Correctness reward.
        else:
            return (
                {"formatted": True},
                0.0,
            )  # Formatted but wrong answer; no format reward to avoid hacking.
    else:
        return _tag_format_rejection(benchmark_result)
