from datasets import load_dataset

dataset = load_dataset("nq_open", split="train[:5]")

for item in dataset:
    print(item["question"])
    print(item["answer"])
    print("-----")