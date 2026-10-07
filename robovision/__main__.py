"""Command line interface for the centered cup pickup project."""

import argparse
import importlib


def main():
    commands = {
        "train-joint": ("train_joint", "main"),
        "evaluate-joint": ("evaluate_joint", "main"),
        "evaluate-final": ("project", "evaluate_final"),
        "report": ("project", "report"),
        "demo": ("project", "demo"),
    }
    parser = argparse.ArgumentParser(
        description="CNN/PPO force control for centered cup pickup"
    )
    parser.add_argument("command", choices=commands)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module, function = commands[args.command]
    getattr(importlib.import_module(f"robovision.{module}"), function)(args.args)


if __name__ == "__main__":
    main()
