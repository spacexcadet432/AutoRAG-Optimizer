"""
CLI script to build a Wikipedia corpus for the NQ-Open subset.
"""

from project_datasets.loader import NQQuery, build_wikipedia_corpus_for_nq, load_nq_queries_from_disk


def main() -> None:
    raw_queries = load_nq_queries_from_disk()
    queries = [NQQuery(question=q["question"], answer=q["answer"]) for q in raw_queries]

    corpus, mapping = build_wikipedia_corpus_for_nq(queries)
    print(f"Built corpus with {len(corpus)} unique articles.")
    print(f"Built mapping for {len(mapping)} questions.")


if __name__ == "__main__":
    main()


