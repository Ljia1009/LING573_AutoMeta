import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Full path of the file used for testing. ",
    )

    parser.add_argument(
        "--data_option",
        type=str,
        default="test",
        help="Option for the file used for testing, ignored when full path is provided. Valid options are dev, test, or train.",
    )
    parser.add_argument(
        "--key_option",
        type=str,
        default="review",
        help="Option for the keys extracted from each review. Valid options are review, all.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output.",
    )

    parser.add_argument(
        "--evaluation_result_path",
        type=str,
        help="Path to save the evaluation result.",
    )
    args = parser.parse_args()
    return args
