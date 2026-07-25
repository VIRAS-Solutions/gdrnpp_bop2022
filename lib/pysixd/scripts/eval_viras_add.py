#!/usr/bin/env python3
"""VIRAS-specific evaluator using ADD metrics (Average Distance of Model Points).

Calculates:
- ADD: Average Distance of Model Points (main metric)
- Rotation Error: Angle between estimated and ground-truth rotation
- Translation Error: L2 distance between translations

Usage:
    python eval_viras_add.py --results_path /path/to/results --dataset_root /path/to/dataset
"""
import argparse
import json
import logging
import os
import os.path as osp
from collections import defaultdict

import numpy as np
from scipy.linalg import logm

from lib.pysixd import inout
from lib.pysixd.pose_error import add, te, re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rotation_error_deg(R_est, R_gt):
    """Calculate rotation error in degrees.
    
    Args:
        R_est: 3x3 estimated rotation matrix
        R_gt: 3x3 ground-truth rotation matrix
        
    Returns:
        Error in degrees
    """
    # Formula: error = arccos((trace(R_est^T @ R_gt) - 1) / 2)
    trace = np.trace(R_est.T @ R_gt)
    # Clamp to [-1, 1] to avoid numerical issues
    trace = np.clip(trace, -1.0, 1.0)
    angle_rad = np.arccos((trace - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def evaluate_viras_results(results_path, dataset_root, obj_id=None):
    """Evaluate VIRAS results using ADD metrics.
    
    Args:
        results_path: Path to results directory containing JSON files
        dataset_root: Root path of VIRAS dataset
        obj_id: Specific object ID to evaluate (optional)
        
    Returns:
        Dictionary with evaluation results
    """
    # Load results
    results_file = None
    for fname in os.listdir(results_path):
        if fname.endswith('.json') and 'viras' in fname.lower():
            results_file = osp.join(results_path, fname)
            break
    
    if results_file is None:
        # Try to find any JSON file
        json_files = [f for f in os.listdir(results_path) if f.endswith('.json')]
        if not json_files:
            logger.error(f"No JSON results found in {results_path}")
            return {}
        results_file = osp.join(results_path, json_files[0])
    
    logger.info(f"Loading results from: {results_file}")
    results = json.load(open(results_file, 'r'))
    
    # Load models
    objects_info_path = osp.join(dataset_root, 'objects_info.json')
    objects_info = json.load(open(objects_info_path, 'r')) if osp.exists(objects_info_path) else {}
    
    models_dict = {}
    models_dir = osp.join(dataset_root, 'models')
    for model_id_str, model_data in objects_info.get('models', {}).items():
        model_id = int(model_id_str)
        local_cat_id = model_data.get('local_category_id', model_id)
        file_id = local_cat_id + 1
        model_path = osp.join(models_dir, f'obj_{file_id:06d}.ply')
        if osp.exists(model_path):
            try:
                model = inout.load_ply(model_path, vertex_scale=0.001)  # Scale to meters
                models_dict[model_id] = model['pts']
                logger.info(f"Loaded model {model_id} from {model_path}: {len(model['pts'])} points")
            except Exception as e:
                logger.warning(f"Failed to load model {model_id}: {e}")
    
    if not models_dict:
        logger.error("No models loaded!")
        return {}
    
    # Evaluate results
    metrics = defaultdict(lambda: {'add_errors': [], 'rot_errors': [], 'trans_errors': []})
    
    for result in results:
        est_obj_id = result.get('obj_id', result.get('category_id'))
        gt_obj_id = result.get('obj_id', result.get('category_id'))
        
        if est_obj_id not in models_dict:
            logger.warning(f"Model {est_obj_id} not found in loaded models")
            continue
        
        # Extract poses
        R_est = np.array(result.get('R_est', result.get('rotation', np.eye(3)))).reshape(3, 3)
        t_est = np.array(result.get('t_est', result.get('translation', np.zeros(3)))).reshape(3, 1)
        
        R_gt = np.array(result.get('R_gt', result.get('rotation_gt', np.eye(3)))).reshape(3, 3)
        t_gt = np.array(result.get('t_gt', result.get('translation_gt', np.zeros(3)))).reshape(3, 1)
        
        # Get model points
        pts = models_dict[est_obj_id]
        
        # Calculate metrics
        try:
            add_error = add(R_est, t_est, R_gt, t_gt, pts)
            rot_error = rotation_error_deg(R_est, R_gt)
            trans_error = te(t_est, t_gt) * 1000  # Convert to mm
            
            metrics[est_obj_id]['add_errors'].append(add_error)
            metrics[est_obj_id]['rot_errors'].append(rot_error)
            metrics[est_obj_id]['trans_errors'].append(trans_error)
        except Exception as e:
            logger.warning(f"Failed to compute metrics for result: {e}")
    
    # Aggregate results
    summary = {}
    all_add = []
    all_rot = []
    all_trans = []
    
    for obj_id, errors in metrics.items():
        if errors['add_errors']:
            mean_add = np.mean(errors['add_errors'])
            mean_rot = np.mean(errors['rot_errors'])
            mean_trans = np.mean(errors['trans_errors'])
            
            summary[f'obj_{obj_id}'] = {
                'add_mean': mean_add,
                'add_std': np.std(errors['add_errors']),
                'add_max': np.max(errors['add_errors']),
                'rot_mean_deg': mean_rot,
                'trans_mean_mm': mean_trans,
                'num_predictions': len(errors['add_errors']),
            }
            
            all_add.extend(errors['add_errors'])
            all_rot.extend(errors['rot_errors'])
            all_trans.extend(errors['trans_errors'])
    
    if all_add:
        summary['overall'] = {
            'add_mean': np.mean(all_add),
            'add_std': np.std(all_add),
            'add_median': np.median(all_add),
            'add_max': np.max(all_add),
            'rot_mean_deg': np.mean(all_rot),
            'rot_std_deg': np.std(all_rot),
            'trans_mean_mm': np.mean(all_trans),
            'trans_std_mm': np.std(all_trans),
            'num_predictions': len(all_add),
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate VIRAS results with ADD metrics')
    parser.add_argument('--results_path', type=str, required=True, help='Path to results directory')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root path of VIRAS dataset')
    parser.add_argument('--obj_id', type=int, default=None, help='Specific object ID to evaluate')
    
    args = parser.parse_args()
    
    results = evaluate_viras_results(args.results_path, args.dataset_root, args.obj_id)
    
    # Print results
    print("\n" + "="*80)
    print("VIRAS ADD Evaluation Results")
    print("="*80)
    
    if 'overall' in results:
        print("\nOVERALL RESULTS:")
        for key, value in results['overall'].items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:10.4f}")
            else:
                print(f"  {key:25s}: {value}")
    
    print("\nPER-OBJECT RESULTS:")
    for obj_name, metrics in sorted(results.items()):
        if obj_name != 'overall':
            print(f"\n  {obj_name}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {key:20s}: {value:10.4f}")
                else:
                    print(f"    {key:20s}: {value}")
    
    print("\n" + "="*80)
    
    # Save summary as JSON
    summary_file = osp.join(args.results_path, 'viras_add_results.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to: {summary_file}")


if __name__ == '__main__':
    main()
