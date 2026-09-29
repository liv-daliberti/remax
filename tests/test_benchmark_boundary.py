"""The trainer must use the installed benchmark's exact admission boundary."""
import pytest


def test_training_grader_delegates_execution_identity():
    pytest.importorskip('math_verify', reason='optional training grader dependencies')
    from remax import math_grader
    from modebench import grading
    assert math_grader.validated_modebench_outcome_key is grading.validated_modebench_outcome_key
    assert math_grader.VerifiedExplorationIdentity is grading.VerifiedExplorationIdentity
    spec = {'verifier':'countdown','numbers':[1,2,3],'target':6}
    assert math_grader.boxed_reward_fn('1+2+3',spec)[1] == 1.0
    assert math_grader.validated_modebench_outcome_key('1+2',spec) is None
