import argparse

from crime_agent.app import build_app, run_demo, run_repl, run_validation_test_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natural language to Neo4j (Gemini + LangChain).")
    parser.add_argument("--demo", action="store_true", help="Run a small demo with sample questions.")
    parser.add_argument(
        "--run-validation-tests",
        action="store_true",
        help="Print 3 validation test cases (required by the assignment).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print routing/Cypher/validation details.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(verbose=args.verbose)

    if args.run_validation_tests:
        run_validation_test_cases(app)
        return

    if args.demo:
        run_demo(app)
        return

    run_repl(app)


if __name__ == "__main__":
    main()
