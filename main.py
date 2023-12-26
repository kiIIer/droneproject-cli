import argparse
from video_splitter import setup_cli


def main():
    parser = argparse.ArgumentParser(description='Video Processing Application')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    setup_cli(subparsers)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
