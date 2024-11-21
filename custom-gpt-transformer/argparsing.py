def parse_args():
    parser = argparse.ArgumentParser(
        description='Infinite in Modern Thought'
    )

    parser.add_argument(
        '-bse',
        type=str,
        required=True,
        help='Please enter the batch size'
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.bs)

if __name__ == '__main__':
    main()