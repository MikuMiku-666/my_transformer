import pickle
import numpy as np

with open(r"dataset/raw/10.pkl", "rb") as f:
    obj = pickle.load(f)

lens = [s["x"].shape[0] for s in obj]
print("frames:", len(lens))
print("min points per frame:", min(lens))
print("max points per frame:", max(lens))
print("mean points per frame:", np.mean(lens))
print("median points per frame:", np.median(lens))
