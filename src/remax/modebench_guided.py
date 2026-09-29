"""Target-blind legal-output grammars for the admitted Level-2 benchmark."""

from __future__ import annotations

import itertools
import json
import re


PROFILES = frozenset({"none", "domain_legal_v1", "countdown_legal_v3"})


def countdown_legal_regex(reference: str) -> str:
    """Enumerate operand-exact expression syntax without consulting the target."""

    numbers = tuple(str(int(value)) for value in json.loads(reference)["numbers"])
    operator = r"[+*/-]"

    def trees(values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) == 1:
            return (re.escape(values[0]),)
        expressions = []
        for split in range(1, len(values)):
            for left in trees(values[:split]):
                for right in trees(values[split:]):
                    expressions.append(r"\(" + left + " " + operator + " " + right + r"\)")
        return tuple(expressions)

    expressions = set()
    for values in set(itertools.permutations(numbers)):
        expressions.update(trees(values))
    return r"\\boxed\{(?:" + "|".join(sorted(expressions)) + r")\}"


def legal_regex(profile: str, domain: str, reference: str) -> str | None:
    """Return the frozen legal syntax for one row; never inspect its target."""

    if profile not in PROFILES:
        raise ValueError(f"unknown ModeBench syntax profile: {profile}")
    if profile == "none":
        return None
    if profile == "countdown_legal_v3":
        if domain != "countdown":
            raise ValueError("countdown_legal_v3 is Countdown-only")
        return countdown_legal_regex(reference)
    if domain == "pantry_plan":
        spec = json.loads(reference)
        alternatives = []
        for ingredient in spec["ingredients"]:
            identifier = str(ingredient["id"])
            minimum = int(ingredient["min_if_used_g"])
            available = int(ingredient["available_g"])
            step = int(ingredient["step_g"])
            alternatives.extend(
                f"{identifier}={grams}"
                for grams in range(minimum, available + 1, step)
            )
        atom = "(?:" + "|".join(re.escape(value) for value in alternatives) + ")"
        return r"\\boxed\{" + atom + "(?:;" + atom + r"){1,3}\}"
    fixed = {
        "python_factors": r"\\boxed\{lambda n: [A-Za-z0-9 _%<>=!+*/().-]+\}",
        "mathir": r"\\boxed\{[A-F](?:;[A-F]){0,3}\}",
    }
    if domain not in fixed:
        raise ValueError(f"domain_legal_v1 has no grammar for {domain}")
    return fixed[domain]


def guided_sampling_params(base, profile: str, domain: str, references: list[str]):
    """Copy one vLLM parameter object per row and attach its legal grammar."""

    if profile == "none":
        return base
    import copy
    from vllm.sampling_params import GuidedDecodingParams

    params = []
    for reference in references:
        row = copy.copy(base)
        regex = legal_regex(profile, domain, str(reference))
        if regex is None:
            raise RuntimeError("active syntax profile resolved no grammar")
        row.guided_decoding = GuidedDecodingParams(regex=regex)
        params.append(row)
    return params
