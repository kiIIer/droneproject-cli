import argparse

import presence_detector
import video_splitter


def main():
    parser = argparse.ArgumentParser(description='Video Processing Application')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    video_splitter.setup_cli(subparsers)
    presence_detector.setup_cli(subparsers)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
