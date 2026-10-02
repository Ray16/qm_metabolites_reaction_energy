"""Command-line entry point for MetaG."""


def main() -> None:
    """Run the frozen pipeline CLI without importing the QM stack at startup."""
    from metag.pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()

