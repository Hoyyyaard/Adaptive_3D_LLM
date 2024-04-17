import os
from collections import Counter
import json
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open("data/ScanQA/ScanQA_v1.0_val.json","r"))
instance_counter = Counter()
total_size = 0

for d in data:
    for obj in d['object_names']:
        instance_counter[obj] += 1
        total_size += 1

# print(total_size)
# print(instance_counter)

# Calculating the total count and converting counts to percentages
total_count = sum(instance_counter.values())

instance_counter = dict(sorted(instance_counter.items(), key=lambda item: item[1], reverse=True))
instance_counter = dict(list(instance_counter.items())[:20])

percentages = {item: (count / total_count) * 100 for item, count in instance_counter.items()}

np.random.seed(0)  # For consistent colors across different runs
colors = np.random.rand(len(instance_counter), 3)

# Plotting with percentages on the y-axis
plt.figure(figsize=(14, 10))
plt.bar(percentages.keys(), percentages.values(), color=colors)
plt.xlabel('Items')
plt.ylabel('Percentage (%)')
plt.title(f'Percentage of Various Items (Total: {total_count})')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
