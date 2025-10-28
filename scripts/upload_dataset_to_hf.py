import argparse
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Upload a DatasetDict to the Hugging Face Hub.")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face API token.")
    parser.add_argument("--repo", type=str, required=True,
                        help="Repository name in the format 'username/dataset_name'.")
    parser.add_argument("--directory", type=str, required=True,
                        help="Path to your dataset file (e.g., CSV, JSON, etc.).")

    args = parser.parse_args()

    # Load the dataset from a file. Here we assume a CSV file;
    # you can change the loading script depending on your file type.
    dataset = load_from_disk(args.directory)

    # Push the dataset to the Hugging Face Hub.
    # The push_to_hub() method automatically creates the repo if it doesn't exist.
    dataset.push_to_hub(
        repo_id=args.repo,
        token=args.token,
        private=True
    )
    print(f"Dataset pushed to the Hub at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
