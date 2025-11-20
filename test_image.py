import pickle
import numpy as np

with open('./data/FB15K-DB15K/clip_image_1.pkl', 'rb') as f:
    data = pickle.load(f)

count_non_zero = 0
for k, v in data.items():
    if np.any(np.array(v) != 0):
        count_non_zero += 1

print(f"Total keys: {len(data)}")
print(f"Non-zero vectors: {count_non_zero}")