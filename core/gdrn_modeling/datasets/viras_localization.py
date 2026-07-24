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
        self.objects_info_path = data_cfg.get("objects_info_path")  # NEW: Path to filtered objects_info.json
        self.category_id_map = None  # Will be loaded from objects_info.json if available

    def __call__(self):
        # Include filter_model_ids in cache hash if specified
        filter_model_ids_str = str(self.data_cfg.get("filter_model_ids", ""))
        hashed = hashlib.md5(
            (
                f"{self.name}|{self.dataset_path}|{self.split}|"
                f"{self.with_masks}|{self.filter_invalid}|{filter_model_ids_str}|{__name__}"
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.cache_dir, f"dataset_dicts_{self.name}_{hashed}.pkl")

        if self.use_cache and osp.exists(cache_path):
            logger.info("load cached dataset dicts from %s", cache_path)
            cached_dicts = mmcv.load(cache_path)
            if not cached_dicts:
                logger.warning("Cached dataset is EMPTY - regenerating from scratch")
                # Delete the poisoned cache file
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            else:
                logger.info("Loaded %d records from cache", len(cached_dicts))
                return cached_dicts

        logger.info("[VIRAS] Generating dataset from scratch for split=%s, dataset_path=%s", self.split, self.dataset_path)
        t_start = time.perf_counter()
        reader = UnifiedDatasetReader(self.dataset_path)
        
        # Load filtered objects_info.json if available (with local_category_id mapping)
        if self.objects_info_path and osp.isfile(self.objects_info_path):
            logger.info("[VIRAS] Loading filtered objects_info from: %s", self.objects_info_path)
            try:
                import json
                with open(self.objects_info_path, "r") as f:
                    filtered_objects_info = json.load(f)
                # Extract category_id_map: {model_id: local_category_id}
                self.category_id_map = {}
                for model_id_str, model_data in filtered_objects_info.get("models", {}).items():
                    model_id = int(model_id_str)
                    local_category_id = model_data.get("local_category_id")
                    if local_category_id is not None:
                        self.category_id_map[model_id] = local_category_id
                logger.info("[VIRAS] Loaded category_id_map from objects_info: %s", self.category_id_map)
            except Exception as e:
                logger.warning("[VIRAS] Failed to load category_id_map from %s: %s", self.objects_info_path, e)
                self.category_id_map = None
        else:
            if self.objects_info_path:
                logger.info("[VIRAS] objects_info_path specified but file not found: %s", self.objects_info_path)
            else:
                logger.info("[VIRAS] No filtered objects_info_path provided - using default adapter mapping")
        
        # Pass filter_model_ids to adapter if specified (for single-object training)
        filter_model_ids = self.data_cfg.get("filter_model_ids")
        logger.info("[VIRAS] Dataset filter_model_ids: %s", filter_model_ids)
        adapter = GdrnppLocalizationAdapter(reader, self.split, filter_model_ids=filter_model_ids)
        logger.info("[VIRAS] Adapter model_names: %s (count=%d)", adapter.model_names, len(adapter.model_names))
        
        # Validate that adapter's mapping matches loaded category_id_map (if available)
        if self.category_id_map:
            logger.info("[VIRAS] Adapter._model_index: %s", adapter._model_index)
            if adapter._model_index == self.category_id_map:
                logger.info("[VIRAS] ✓ category_id_map matches adapter mapping - consistent!")
            else:
                logger.warning("[VIRAS] ⚠ category_id_map mismatch! Expected: %s, got: %s", 
                              self.category_id_map, adapter._model_index)

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
                        "category_id": inst.get("local_category_id", inst["category_id"]),
                        "root_category_id": inst["category_id"],  # die ursprüngliche Kategorie-ID
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

        logger.info("[VIRAS] loaded %d records in %.2fs", len(dataset_dicts), time.perf_counter() - t_start)
        
        # Only cache if we have data - avoid saving empty cache files
        if dataset_dicts:
            mmcv.mkdir_or_exist(osp.dirname(cache_path))
            mmcv.dump(dataset_dicts, cache_path, protocol=4)
            logger.info("[VIRAS] Cached %d records to %s", len(dataset_dicts), cache_path)
        else:
            logger.warning("[VIRAS] Dataset is empty - NOT caching to avoid poisoning future runs")
        
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
    logger.info(f"[DEBUG register_with_name_cfg] Registering {name} with data_cfg={data_cfg}")
    
    # Remove old registration if it exists (to avoid conflicts with old metadata)
    if name in DatasetCatalog.list():
        logger.info(f"[DEBUG register_with_name_cfg] Dataset {name} already registered, unregistering old version")
        # We can't directly unregister, but we can replace the registration
    
    if name in SPLITS_DATA:
        used_cfg = dict(SPLITS_DATA[name])
        if data_cfg is not None:
            used_cfg.update(data_cfg)
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg

    logger.info(f"[DEBUG register_with_name_cfg] used_cfg keys: {list(used_cfg.keys())}")
    logger.info(f"[DEBUG register_with_name_cfg] filter_model_ids in used_cfg: {used_cfg.get('filter_model_ids')}")
    logger.info(f"[DEBUG register_with_name_cfg] objects_info_path in used_cfg: {used_cfg.get('objects_info_path')}")
    
    reader = UnifiedDatasetReader(used_cfg["dataset_path"])
    # Pass filter_model_ids to adapter if specified (for single-object training)
    filter_model_ids = used_cfg.get("filter_model_ids")
    logger.info(f"[DEBUG register_with_name_cfg] Creating adapter with filter_model_ids={filter_model_ids}")
    adapter = GdrnppLocalizationAdapter(reader, used_cfg["split"], filter_model_ids=filter_model_ids)
    logger.info(f"[DEBUG register_with_name_cfg] Adapter model_names: {adapter.model_names} (count={len(adapter.model_names)})")
    
    # Load and validate category_id_map from filtered objects_info if available
    objects_info_path = used_cfg.get("objects_info_path")
    if objects_info_path and osp.isfile(objects_info_path):
        logger.info(f"[DEBUG register_with_name_cfg] Loading category_id_map from: {objects_info_path}")
        try:
            import json
            with open(objects_info_path, "r") as f:
                filtered_objects_info = json.load(f)
            category_id_map = {}
            for model_id_str, model_data in filtered_objects_info.get("models", {}).items():
                model_id = int(model_id_str)
                local_category_id = model_data.get("local_category_id")
                if local_category_id is not None:
                    category_id_map[model_id] = local_category_id
            logger.info(f"[DEBUG register_with_name_cfg] Loaded category_id_map: {category_id_map}")
            if adapter._model_index == category_id_map:
                logger.info(f"[DEBUG register_with_name_cfg] ✓ Adapter mapping matches objects_info mapping")
            else:
                logger.warning(f"[DEBUG register_with_name_cfg] ⚠ Mapping mismatch! Adapter: {adapter._model_index}, objects_info: {category_id_map}")
        except Exception as e:
            logger.warning(f"[DEBUG register_with_name_cfg] Failed to load category_id_map: {e}")
    
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