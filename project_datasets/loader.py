import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
import requests


NQ_DATA_DIR = os.path.join("datasets", "nq_open")
NQ_QUERIES_PATH = os.path.join(NQ_DATA_DIR, "queries.json")
NQ_CORPUS_PATH = os.path.join(NQ_DATA_DIR, "corpus.json")
NQ_MAPPING_PATH = os.path.join(NQ_DATA_DIR, "mapping.json")


DEFAULT_NQ_SPLIT = "train"
DEFAULT_NQ_NUM_SAMPLES = 500
DEFAULT_SEED = 42


@dataclass
class NQQuery:
    question: str
    answer: str


@dataclass
class NQCorpusItem:
    doc_id: str
    title: str
    text: str


def _ensure_nq_dirs() -> None:
    os.makedirs(NQ_DATA_DIR, exist_ok=True)


def load_nq_dataset(split: str = DEFAULT_NQ_SPLIT) -> Dataset:
    return load_dataset("nq_open", split=split)


def build_nq_query_subset(
    num_samples: int = DEFAULT_NQ_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    split: str = DEFAULT_NQ_SPLIT,
) -> List[NQQuery]:
    _ensure_nq_dirs()

    ds = load_nq_dataset(split=split)
    rng = random.Random(seed)

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    subset: List[NQQuery] = []

    for idx in indices:
        if len(subset) >= num_samples:
            break

        item = ds[int(idx)]
        answers = item.get("answer") or []
        if not answers:
            continue

        subset.append(NQQuery(question=item["question"], answer=answers[0]))

    to_dump = [
        {"id": i, "question": q.question, "answer": q.answer}
        for i, q in enumerate(subset)
    ]

    with open(NQ_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(to_dump, f, indent=2, ensure_ascii=False)

    return subset


def _wikipedia_search(query: str, session: requests.Session) -> Tuple[str, str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }

    resp = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("query", {}).get("search", [])
    if not hits:
        raise RuntimeError("No Wikipedia search results")

    top = hits[0]
    return str(top["pageid"]), top["title"]


def _wikipedia_page_text(pageid: str, session: requests.Session) -> str:
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }

    resp = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    page = pages.get(pageid)
    if not page:
        raise RuntimeError("Wikipedia page not found")

    text = page.get("extract") or ""
    return text.strip()


def build_wikipedia_corpus_for_nq(
    queries: List[NQQuery],
) -> Tuple[List[NQCorpusItem], Dict[int, str]]:
    _ensure_nq_dirs()

    session = requests.Session()

    corpus: Dict[str, NQCorpusItem] = {}
    question_to_doc: Dict[int, str] = {}

    for q_idx, q in enumerate(queries):
        try:
            pageid, title = _wikipedia_search(q.question, session=session)
            if pageid not in corpus:
                text = _wikipedia_page_text(pageid, session=session)
                if not text:
                    continue

                corpus[pageid] = NQCorpusItem(
                    doc_id=pageid,
                    title=title,
                    text=text,
                )

            question_to_doc[q_idx] = pageid
        except Exception:
            continue

    corpus_list = [
        {"doc_id": item.doc_id, "title": item.title, "text": item.text}
        for item in corpus.values()
    ]

    with open(NQ_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_list, f, indent=2, ensure_ascii=False)

    with open(NQ_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(question_to_doc, f, indent=2, ensure_ascii=False)

    return corpus_list, question_to_doc


def load_nq_queries_from_disk() -> List[Dict[str, str]]:
    with open(NQ_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nq_corpus_from_disk() -> List[Dict[str, str]]:
    with open(NQ_CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nq_mapping_from_disk() -> Dict[str, str]:
    with open(NQ_MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

