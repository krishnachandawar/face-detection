import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from face_utils import FACE_SIZE, create_recognizer

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_DIR = BASE_DIR / "trainer"
NAMES_PATH = TRAINER_DIR / "names.json"
MODEL_PATH = TRAINER_DIR / "trainer.yml"
MAX_PER_PERSON = 80

TRAINER_DIR.mkdir(exist_ok=True)

image_paths = sorted(DATASET_DIR.glob("User.*.jpg"))
if not image_paths:
    raise SystemExit(f"No photos found in {DATASET_DIR}. Run 01_register.py first.")

names = {}
if NAMES_PATH.exists():
    names = {int(k): v for k, v in json.loads(NAMES_PATH.read_text(encoding="utf-8-sig")).items()}

by_id = defaultdict(list)
for path in image_paths:
    parts = path.stem.split(".")
    if len(parts) != 3:
        continue
    by_id[int(parts[1])].append(path)

name_to_id = {}
id_remap = {}
next_id = 1
for user_id in sorted(by_id):
    label = names.get(user_id, f"ID {user_id}")
    if label not in name_to_id:
        name_to_id[label] = next_id
        next_id += 1
    id_remap[user_id] = name_to_id[label]

merged_names = {str(v): k for k, v in name_to_id.items()}
NAMES_PATH.write_text(json.dumps(merged_names, indent=2), encoding="utf-8")

faces = []
ids = []
for user_id, paths in sorted(by_id.items()):
    train_id = id_remap[user_id]
    if len(paths) > MAX_PER_PERSON:
        step = max(1, len(paths) // MAX_PER_PERSON)
        paths = paths[::step][:MAX_PER_PERSON]
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        faces.append(cv2.resize(image, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA))
        ids.append(train_id)

if not faces:
    raise SystemExit("Could not load any valid face images.")

unique_ids = sorted(set(ids))
if len(unique_ids) < 2:
    print("Warning: only one person is registered. Add another person for stronger identity separation.")

print(f"Training on {len(faces)} images from {len(unique_ids)} people...")
recognizer = create_recognizer()
recognizer.train(faces, np.array(ids, dtype=np.int32))
recognizer.write(str(MODEL_PATH))

print(f"Trained on {len(faces)} samples from {len(unique_ids)} people.")
print("LBPH settings: radius=1, neighbors=8, grid=8x8.")
print(f"Model saved to {MODEL_PATH}")
print("Registered names:", ", ".join(f"{k}={v}" for k, v in merged_names.items()))
print("Next: python 03_recognize.py")
