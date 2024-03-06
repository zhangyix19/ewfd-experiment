import argparse


class DatasetWithDefenses(argparse.Action):
    def __call__(self, parser, namespace, values, opts, **kwargs):
        lst = getattr(namespace, self.dest)
        if lst is None:
            lst = []
            setattr(namespace, self.dest, lst)

        lst.append((values[0], values[1:] if len(values) > 1 else ["nodef"]))


def trainparser():
    parser = argparse.ArgumentParser(description="WFP Experiment")
    parser.add_argument("-g", "--gpu", default="0", type=str, help="Device id")
    parser.add_argument("-l", "--length", default=10000, type=int, help="length of features")
    parser.add_argument("-e", "--epoch", default=0, type=int, help="epoch of training")
    parser.add_argument(
        "--train",
        default=["undefend"],
        nargs="+",
        type=str,
        help="defense method for training",
    )
    parser.add_argument("--attack", default="RF", type=str, help="attack method")
    parser.add_argument("-d", "--dataset", default="undefend", type=str, help="name of dataset")
    parser.add_argument("-n", "--note", default="test", type=str, help="note of experiment")
    parser.add_argument(
        "--test", default=[], nargs="*", type=str, help="defense method for testing"
    )
    parser.add_argument("--batch_size", default=0, type=int, help="batch size")
    parser.add_argument("--cw_size", default=[100, 100], type=int, nargs="+", help="batch size")
    parser.add_argument("--dump", action="store_true", help="dump dataset")

    return parser


def parse_taskname(args):
    train_defenses = "&".join(args.train)
    train_desc = f"train_{args.dataset}_d{train_defenses}"
    return train_desc


def evaluate_parser():
    parser = argparse.ArgumentParser(description="WFP Experiment Test")
    parser.add_argument("-g", "--gpu", default="9", type=str, help="Device id")
    parser.add_argument("-l", "--length", default=10000, type=int, help="length of features")
    parser.add_argument("--note", default=None, type=str, help="train note")
    parser.add_argument(
        "-d", "--dataset", default="undefend", type=str, help="dataset name for testing"
    )
    parser.add_argument("--train", default="undefend", type=str, help="defense method for training")
    parser.add_argument("--test", default="", type=str, help="defense method for testing")
    parser.add_argument("--attack", default="RF", type=str, help="attack method")
    parser.add_argument("-e", "--epoch", default=None, type=int, help="epoch of model")
    return parser


def shap_parser():
    parser = argparse.ArgumentParser(description="WFP Experiment SHAP Analysis")
    parser.add_argument("-g", "--gpu", default="9", type=str, help="Device id")
    parser.add_argument("-l", "--length", default=10000, type=int, help="length of features")
    parser.add_argument("--note", default=None, type=str, help="train note")
    parser.add_argument(
        "-d", "--dataset", default="undefend", type=str, help="dataset name for testing"
    )
    parser.add_argument("--train", default="undefend", type=str, help="defense method for training")
    parser.add_argument("--attack", default="RF", type=str, help="attack method")
    parser.add_argument("-e", "--epoch", default=None, type=int, help="epoch of model")
    return parser
