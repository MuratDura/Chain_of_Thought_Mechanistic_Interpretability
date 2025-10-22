import argparse
from svamp_dataset import load_svamp_two_operation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter SVAMP to exactly two-operator equations and optionally save."
    )
    parser.add_argument(
        "--json",
        default="SVAMP.json",
        help="Path to SVAMP.json (default: SVAMP.json)",
    )
    parser.add_argument(
        "--save",
        default="SVAMP_two_ops.json",
        help="Output path for filtered dataset (default: SVAMP_two_ops.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write output file; only print count",
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print a small sample item (ID and Equation)",
    )
    args = parser.parse_args()

    items = load_svamp_two_operation(args.json, save_to=None if args.no_save else args.save)
    print(f"two_ops_count={len(items)}")
    if not args.no_save:
        print(f"saved_to={args.save}")
    if args.show_sample and items:
        sample = items[0]
        print(f"sample_id={sample.get('ID')} equation={sample.get('Equation')}")


if __name__ == "__main__":
    main()


