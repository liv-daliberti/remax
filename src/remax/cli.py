"""Small installed-package workflows; training imports remain opt-in."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform

from . import __version__


def demo():
    """An illustrative score-space update, requiring neither weights nor a GPU."""
    import torch
    from .core import (
        binary_maxrl_advantages,
        canonical_replay_uniform_verified_likelihood_loss,
    )

    advantages = binary_maxrl_advantages(torch.tensor([[1.0, 0.0]]))
    scores = torch.tensor([-1.0, -3.0, -2.0], requires_grad=True)
    result = canonical_replay_uniform_verified_likelihood_loss(scores, [2, 1])
    result.loss.backward()
    return {
        "maxrl_advantages": advantages.tolist(),
        "replay_loss": result.loss.item(),
        "score_gradients": scores.grad.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run the CPU replay loss/gradient example")
    recipes = commands.add_parser(
        "recipes", help="list bundled recipes or print one as JSON"
    )
    recipes.add_argument("name", nargs="?")
    environment = commands.add_parser(
        "environment", help="show installed core versions"
    )
    environment.add_argument(
        "--training",
        action="store_true",
        help="also check the qualified training dependency set",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            print(json.dumps(demo()))
        elif args.command == "recipes":
            from .launcher import recipe_names, resolve_recipe

            if args.name:
                print(resolve_recipe(args.name).read_text(), end="")
            else:
                print("\n".join(recipe_names()))
        else:
            versions = {
                name: metadata.version(name)
                for name in ("remax-rl", "modebench", "torch", "numpy")
            }
            if args.training:
                from .launch_record import validate_runtime

                versions.update(validate_runtime())
            print(
                json.dumps(
                    {
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "dependencies": versions,
                    },
                    indent=2,
                )
            )
    except (ValueError, OSError) as error:
        parser.error(str(error))
