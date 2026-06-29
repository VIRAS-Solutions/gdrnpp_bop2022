import hashlib
import logging
import os
import os.path as osp
import sys
import time

import mmcv
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from transforms3d.quaternions import mat2quat

from lib.utils.mask_utils import binary_mask_to_rle
from lib.pysixd import misc
import ref


logger = logging.getLogger(__name__)


def _ensure_dataset_reader_path() -> None:
    """Make dataset_reader_adapter importable in both local and container runs."""
    candidates = []

    env_parent = os.environ.get("VIRAS_DATASET_READER_PARENT")
    if env_parent:
        candidates.append(env_parent)

    env_pkg = os.environ.get("VIRAS_DATASET_READER_PATH")
    if env_pkg:
        candidates.append(osp.dirname(env_pkg.rstrip("/")))

    candidates.extend(
        [
            "/workspace",
            "/workspace/src",
            osp.normpath(osp.join(osp.dirname(__file__), "../../../../../../")),
        ]
    )

    for path in candidates:
        if path and osp.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_dataset_reader_path()

from dataset_reader_adapter.base.unified_dataset_reader import UnifiedDatasetReader  # noqa: E402
from dataset_reader_adapter.localization.gdrnpp_adapter import GdrnppLocalizationAdapter  # noqa: E402


__all__ = ["register_with_name_cfg", "get_available_datasets"]


SPLITS_DATA = {
    "viras_train": {"name": "viras_train", "split": "train"},
    "viras_val": {"name": "viras_val", "split": "val"},
    "viras_test": {"name": "viras_test", "split": "test"},
}


class VIRASDataset:
    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg
        self.dataset_path = data_cfg["dataset_path"]
        self.split = data_cfg["split"]
        self.with_masks = bool(data_cfg.get("with_masks", True))
        self.use_cache = bool(data_cfg.get("use_cache", True))
        self.filter_invalid = bool(data_cfg.get("filter_invalid", True))
        self.cache_dir = data_cfg.get("cache_dir", "/tmp/gdrnpp_cache")

    def __call__(self):
        hashed = hashlib.md5(
            (
                f"{self.name}|{self.dataset_path}|{self.split}|"
                f"{self.with_masks}|{self.filter_invalid}|{__name__}"
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, f"dataset_dicts_{self.name}_{hashed}.pkl")

        if self.use_cache and osp.exists(cache_path):
            logger.info("load cached dataset dicts from %s", cache_path)
            return mmcv.load(cache_path)

        t_start = time.perf_counter()
        reader = UnifiedDatasetReader(self.dataset_path)
        adapter = GdrnppLocalizationAdapter(reader, self.split)

        dataset_dicts = []
        for rec in adapter.iter_gdrn_records(
            dataset_name=self.name,
            with_masks=self.with_masks,
            filter_invalid=self.filter_invalid,
        ):
            converted_annotations = []
            for inst in rec["annotations"]:
                rotation = inst["rotation"]
                translation = inst["translation"]

                pose = np.hstack([rotation, translation.reshape(3, 1)]).astype("float32")
                quat = mat2quat(rotation).astype("float32")

                segm = _mask_to_rle(inst.get("mask"), rec["height"], rec["width"], inst["bbox"])
                if self.filter_invalid and segm is None:
                    continue

                converted_annotations.append(
                    {
                        "category_id": inst["category_id"],
                        "bbox": inst["bbox"],
                        "bbox_obj": inst.get("bbox_obj", inst["bbox"]),
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": translation.astype("float32"),
                        "centroid_2d": np.array(inst["centroid_2d"], dtype=np.float32),
                        "segmentation": segm,
                        "mask_full": segm,
                        "visib_fract": float(inst.get("visib_fract", 1.0)),
                        "model_info": inst["model_info"],
                        "bbox3d_and_center": inst["bbox3d_and_center"],
                    }
                )

            if converted_annotations:
                rec["annotations"] = converted_annotations
                dataset_dicts.append(rec)

        logger.info("loaded %d VIRAS records in %.2fs", len(dataset_dicts), time.perf_counter() - t_start)
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        return dataset_dicts


def _mask_to_rle(mask, im_h: int, im_w: int, bbox):
    if mask is not None:
        return binary_mask_to_rle(mask.astype(bool), compressed=True)

    x, y, w, h = [int(v) for v in bbox]
    if w <= 1 or h <= 1:
        return None

    box_mask = np.zeros((im_h, im_w), dtype=bool)
    x2 = min(x + w, im_w)
    y2 = min(y + h, im_h)
    if x2 <= x or y2 <= y:
        return None
    box_mask[max(y, 0):y2, max(x, 0):x2] = True
    return binary_mask_to_rle(box_mask, compressed=True)


def get_viras_metadata(obj_names, ref_key):
    data_ref = ref.__dict__[ref_key]
    loaded_models_info = data_ref.get_models_info()

    cur_sym_infos = {}
    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info

    return {"thing_classes": obj_names, "sym_infos": cur_sym_infos}


def register_with_name_cfg(name, data_cfg=None):
    print(f"Registering dataset {name} with data_cfg: {data_cfg}")
    if name in SPLITS_DATA:
        used_cfg = dict(SPLITS_DATA[name])
        if data_cfg is not None:
            used_cfg.update(data_cfg)
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg

    reader = UnifiedDatasetReader(used_cfg["dataset_path"])
    adapter = GdrnppLocalizationAdapter(reader, used_cfg["split"])
    ref_key = used_cfg.get("ref_key", "viras")

    DatasetCatalog.register(name, VIRASDataset(used_cfg))
    MetadataCatalog.get(name).set(
        id=ref_key,
        ref_key=ref_key,
        objs=adapter.model_names,
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_viras_metadata(obj_names=adapter.model_names, ref_key=ref_key),
    )


def get_available_datasets():
    return list(SPLITS_DATA.keys())