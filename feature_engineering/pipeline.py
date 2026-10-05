"""Generate compact Home Credit features and optional shared folds."""
import argparse
from .io import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='raw_data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--days-per-month', type=float, default=30)
    parser.add_argument('--nested-folds', action='store_true',
                        help='Also export shared 10-outer/3-inner stratified folds')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
