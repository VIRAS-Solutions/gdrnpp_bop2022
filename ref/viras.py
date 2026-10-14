# encoding: utf-8
"""Runtime reference for VIRAS datasets in GDRNPP.

This module is intentionally environment-driven to avoid hardcoded dataset names.
Required env vars:
    VIRAS_GDRN_DATASET_PATH
Optional env vars:
    VIRAS_GDRN_MODELS_ROOT
    VIRAS_GDRN_VERTEX_SCALE
"""

import os
import os.path as osp

import mmcv
import numpy as np

from pathlib import Path


cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")

dataset_root = os.environ.get("VIRAS_GDRN_DATASET_PATH", "")
if not dataset_root:
    raise RuntimeError("VIRAS_GDRN_DATASET_PATH is required for ref.viras")


data_root = osp.dirname(dataset_root)
model_dir = os.environ.get("VIRAS_GDRN_MODELS_ROOT", osp.join(dataset_root, "models"))
model_eval_dir = model_dir  # For evaluation, use the same models as for training (prepared_models with filtering applied)
vertex_scale = float(os.environ.get("VIRAS_GDRN_VERTEX_SCALE", "1.0"))
extra_model_roots = [p for p in os.environ.get("VIRAS_GDRN_MODEL_SEARCH_ROOTS", "").split(":") if p]

objects_info_path = osp.join(dataset_root, "objects_info.json")
models_info_path = osp.join(dataset_root, "models_info.json")
camera_info_path = osp.join(dataset_root, "camera_info.json")

# Check if there's a filtered objects_info.json in the model directory (from validate_and_prepare)
# This is used when training with only a subset of objects
filtered_objects_info_path = osp.join(os.path.dirname(model_dir), "objects_info.json")
if osp.exists(filtered_objects_info_path):
    print(f"[DIAG viras.py] Using filtered objects_info.json from: {filtered_objects_info_path}")
    objects_info_path = filtered_objects_info_path
else:
    print(f"[DIAG viras.py] Using global objects_info.json from: {objects_info_path}")

assert osp.exists(objects_info_path), objects_info_path
assert osp.exists(models_info_path), models_info_path
assert osp.exists(camera_info_path), camera_info_path

objects_info = mmcv.load(objects_info_path)
models_info = mmcv.load(models_info_path)
camera_info = mmcv.load(camera_info_path)

raw_models = objects_info.get("objects", {})
id2obj = {
    int(model_id): model_info.get("display_name") or model_info.get("name") or str(model_id)
    for model_id, model_info in raw_models.items()
}
id2obj = dict(sorted(id2obj.items(), key=lambda kv: kv[0]))

objects = list(id2obj.values())
obj2id = {_name: _id for _id, _name in id2obj.items()}

obj_num = len(objects)

# Build model paths using position index + 1 for file naming (1-indexed)
# Store mapping for later use in renderer and other places
obj_id_to_file_id = {}  # maps global obj_id → file_id (1-indexed)
model_paths = []
texture_paths = []
for idx, obj_id in enumerate(id2obj):
    file_id = idx + 1  # Position in sorted list + 1 = 1-indexed file_id (0, 1, 2, ... → 1, 2, 3, ...)
    obj_id_to_file_id[obj_id] = file_id
    model_paths.append(osp.join(model_dir, f"obj_{file_id:06d}.ply"))
    #texture_paths.append(osp.join(model_dir, f"obj_{file_id:06d}.png"))
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]

diameters = np.array([models_info.get("diameter", 0.0) for obj_id in id2obj], dtype=np.float32)

first_cam = next(iter(camera_info.values()))
width = int(first_cam.get("width", 640))
height = int(first_cam.get("height", 480))
camera_matrix = np.array(first_cam.get("cam_K", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])).reshape(3, 3)

zNear = 0.01
zFar = 100.0
center = (height / 2, width / 2)
depth_factor = 1000.0


def _iter_search_roots():
    roots = [model_dir, osp.join(dataset_root, "models"), dataset_root, data_root, osp.join(data_root, "objects")]
    roots.extend(extra_model_roots)
    seen = set()
    for root in roots:
        if not root:
            continue
        norm = osp.normpath(root)
        if norm in seen:
            continue
        seen.add(norm)
        yield norm


def get_model_path(obj_id: int) -> str:

    canonical = osp.join(model_dir, f"obj_{obj_id_to_file_id[obj_id]:06d}.ply")
    if osp.exists(canonical):
        return canonical

    model_meta = objects_info.get("objects", {}).get(str(obj_id), {})
    rel_model = model_meta.get("files", {}).get("objects") if isinstance(model_meta, dict) else None
    object_path = model_meta.get("object_path") if isinstance(model_meta, dict) else None
    explicit_path = model_meta.get("model_path") if isinstance(model_meta, dict) else None

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    for root in _iter_search_roots():
        candidates.append(osp.join(root, f"obj_{obj_id_to_file_id[obj_id]:06d}.ply"))
        candidates.append(osp.join(root, "prepared_models", f"obj_{obj_id_to_file_id[obj_id]:06d}.ply"))
        if rel_model:
            candidates.append(osp.join(root, rel_model))
            if object_path:
                candidates.append(osp.join(root, object_path, rel_model))
                candidates.append(osp.join(root, "objects", object_path, rel_model))

    for candidate in candidates:
        if osp.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No model file found for model_id {obj_id} - {obj_id_to_file_id[obj_id]}. "
        f"Expected canonical path {canonical} or matching objects_info paths."
    )


model_paths = [get_model_path(obj_id) for obj_id in id2obj]


def get_models_info(obj_folder_name: str=None):
    if obj_folder_name is not None:
        return mmcv.load(osp.join(Path(models_info_path).parent, obj_folder_name, "model_info.json"))
    return mmcv.load(models_info_path)


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    return mmcv.load(fps_points_path)


def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    return mmcv.load(keypoints_3d_path)