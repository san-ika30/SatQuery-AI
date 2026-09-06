import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "backend"))

from datasets.cdvqa import CDVQADataset
from core.change_analyzer import get_change_analyzer

dataset = CDVQADataset()
analyzer = get_change_analyzer()
correct = 0
total = 0

for i in range(len(dataset)):
    item = dataset[i]
    if item.get("change_category") == "change_to_what":
        t1, t2 = dataset.load_images(i)
        res = analyzer.analyze_change(t1, t2, item["question"])
        ans = res.get("direct_answer", "")
        gt = item["ground_truth"]
        match = (ans.strip().lower() == gt.strip().lower()) or (gt.lower() in res.get("natural_answer", "").lower())
        total += 1
        if match:
            correct += 1
        print(f"{item['sample_id']}: GT={gt}, direct={ans}, match={match}")

print(f"Total: {total}, Correct: {correct}, Acc: {correct/total*100:.2f}%")
