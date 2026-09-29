"""Independent scalar reference: no ReMax imports, torch, or autograd.

For the one-epoch same-snapshot PPO cases, ratio=1 and clipping is inactive.
Fresh Dr.GRPO uses r-mean(r); MaxRL uses N*r/K-1. Both divide token sums
by Tmax. Replay averages response log-probabilities, then modes, then banks,
with coefficient alpha*(N-1)/N**2. Differentiate the categorical softmax
explicitly for every bigram, including EOS. SGD is theta-lr*gradient.
"""

import math


def reference_update(table, spec, step, groups, *, method, alpha=None):
    width = spec["num_samples"]
    temperature = spec["temperature"]
    probabilities = []
    for row in table:
        maximum = max(row)
        exp = [math.exp((x - maximum) / temperature) for x in row]
        probabilities.append([x / sum(exp) for x in exp])
    fresh = [[0.0] * len(table[0]) for _ in table]
    replay = [[0.0] * len(table[0]) for _ in table]

    def accumulate(target, prompt, response, coefficient):
        previous = prompt[-1]
        score = 0.0
        for token in response:
            p = probabilities[previous]
            score += math.log(p[token])
            for column, probability in enumerate(p):
                target[previous][column] += (
                    coefficient * ((column == token) - probability) / temperature
                )
            previous = token
        return score / len(response)

    rewards = step["rewards"]
    if method in ("maxrl", "remax"):
        advantages = [
            width * r / sum(rewards) - 1 if sum(rewards) else 0.0 for r in rewards
        ]
    else:
        advantages = [r - sum(rewards) / width for r in rewards]
    for response, advantage, active in zip(
        step["responses"], advantages, step["active"]
    ):
        accumulate(
            fresh,
            step["prompt"],
            response,
            -advantage * active / (width * spec["max_response_length"]),
        )
    alpha = spec["alpha"] if alpha is None else alpha
    coefficient = alpha * (width - 1) / (width * width)
    scores = []
    loss = 0.0
    for group in groups:
        responses = group["response_token_ids"]
        for response in responses:
            row_score = accumulate(
                replay,
                group["prompt_token_ids"],
                response,
                -coefficient / (len(groups) * len(responses) * len(response)),
            )
            scores.append(row_score)
            loss -= row_score / (len(groups) * len(responses))
    if method in ("maxrl", "drgrpo"):
        replay = [[0.0] * len(table[0]) for _ in table]
    gradient = [[f + r for f, r in zip(fr, rr)] for fr, rr in zip(fresh, replay)]
    parameters = [
        [p - spec["learning_rate"] * g for p, g in zip(pr, gr)]
        for pr, gr in zip(table, gradient)
    ]
    return dict(
        fresh=fresh,
        replay=replay,
        gradient=gradient,
        parameters=parameters,
        scores=scores,
        loss=loss,
        raw_weighted_loss=loss * coefficient,
    )
