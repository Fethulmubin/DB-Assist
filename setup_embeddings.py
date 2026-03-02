from crime_agent.embedding_setup import parse_args, run_embedding_setup


def main() -> None:
    config = parse_args()
    run_embedding_setup(config)


if __name__ == "__main__":
    main()
