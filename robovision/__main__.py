"""Command line interface for training, evaluation, and playback."""
import argparse
import importlib


def main():
    commands = {
        "demo": ("demo", []),
        "evaluate": ("evaluate", []),
        "train-target": ("train_target", []),
        "collect": ("train_grasp", ["collect"]),
        "train-grasp": ("train_grasp", ["train"]),
        "validate-grasp": ("train_grasp", ["validate"]),
    }
    parser = argparse.ArgumentParser(description="Vision-guided cup grasping")
    parser.add_argument("command", choices=commands)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module, prefix = commands[args.command]
    importlib.import_module(f"robovision.{module}").main(prefix + args.args)


if __name__ == "__main__":
    main()
