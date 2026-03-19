"""
CLI script to prepare a reproducible NQ-Open query subset.
"""

from project_datasets.loader import DEFAULT_NQ_NUM_SAMPLES, build_nq_query_subset


def main() -> None:
    subset = build_nq_query_subset(num_samples=DEFAULT_NQ_NUM_SAMPLES)
    print(f"Collected {len(subset)} queries and saved to datasets/nq_open/queries.json")


if __name__ == "__main__":
    main()

