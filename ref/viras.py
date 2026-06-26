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


cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")

dataset_root = os.environ.get("VIRAS_GDRN_DATASET_PATH", "")
if not dataset_root:
    raise RuntimeError("VIRAS_GDRN_DATASET_PATH is required for ref.viras")

data_root = osp.dirname(dataset_root)
model_dir = os.environ.get("VIRAS_GDRN_MODELS_ROOT", osp.join(dataset_root, "models"))
vertex_scale = float(os.environ.get("VIRAS_GDRN_VERTEX_SCALE", "1.0"))
extra_model_roots = [p for p in os.environ.get("VIRAS_GDRN_MODEL_SEARCH_ROOTS", "").split(":") if p]

objects_info_path = osp.join(dataset_root, "objects_info.json")
models_info_path = osp.join(dataset_root, "models_info.json")
camera_info_path = osp.join(dataset_root, "camera_info.json")

assert osp.exists(objects_info_path), objects_info_path
assert osp.exists(models_info_path), models_info_path
assert osp.exists(camera_info_path), camera_info_path

objects_info = mmcv.load(objects_info_path)
models_info = mmcv.load(models_info_path)
camera_info = mmcv.load(camera_info_path)

raw_models = objects_info.get("models", {})
id2obj = {
    int(model_id): model_info.get("display_name") or model_info.get("name") or str(model_id)
    for model_id, model_info in raw_models.items()
}
id2obj = dict(sorted(id2obj.items(), key=lambda kv: kv[0]))

objects = list(id2obj.values())
obj2id = {_name: _id for _id, _name in id2obj.items()}
obj_num = len(objects)

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]
texture_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.png") for obj_id in id2obj]
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]

diameters = np.array([models_info[str(obj_id)].get("diameter", 0.0) for obj_id in id2obj], dtype=np.float32)

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
    canonical = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
    if osp.exists(canonical):
        return canonical

    model_meta = objects_info.get("models", {}).get(str(obj_id), {})
    rel_model = model_meta.get("files", {}).get("model") if isinstance(model_meta, dict) else None
    object_path = model_meta.get("object_path") if isinstance(model_meta, dict) else None
    explicit_path = model_meta.get("model_path") if isinstance(model_meta, dict) else None

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    for root in _iter_search_roots():
        candidates.append(osp.join(root, f"obj_{obj_id:06d}.ply"))
        candidates.append(osp.join(root, "models", f"obj_{obj_id:06d}.ply"))
        if rel_model:
            candidates.append(osp.join(root, rel_model))
            if object_path:
                candidates.append(osp.join(root, object_path, rel_model))
                candidates.append(osp.join(root, "objects", object_path, rel_model))

    for candidate in candidates:
        if osp.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No model file found for model_id {obj_id}. "
        f"Expected canonical path {canonical} or matching objects_info paths."
    )


model_paths = [get_model_path(obj_id) for obj_id in id2obj]


def get_models_info():
    return mmcv.load(models_info_path)


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    return mmcv.load(fps_points_path)


def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    return mmcv.load(keypoints_3d_path)