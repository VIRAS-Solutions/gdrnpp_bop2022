# inference with detector, gdrn, and refiner
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)


from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os

import cv2
import numpy as np
import torch
import utils

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".tif"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names


def test1():
    image_paths = get_image_list(osp.join(PROJ_ROOT,"datasets/test/rgb2"), osp.join(PROJ_ROOT,"datasets/test/depth2"))
    yolo_predictor = YoloPredictor(
                       exp_name="yolox-x",
                       config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_itodd.py"),
                       ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_itodd_pbr_itodd_bop_test/model_final.pth"),
                       fuse=True,
                       fp16=False
                     )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/itodd_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_itodd.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/itodd_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_itodd/model_final_wo_optim.pth"),
        #camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/itodd/camera.json"),
        path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/itodd/models")
    )

    for cnt, (rgb_img, depth_img) in enumerate(image_paths):
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        outputs = yolo_predictor.inference(image=rgb_img)
        #for output in outputs:
        #	print(output.cpu())
        #	print(rgb_img.shape)
        #	yolo_predictor.visual_yolo(output,rgb_img,["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28"])
        #print("#OUTPUTS#")
        #print(outputs)
        
        data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=depth_img)
        
        print("\n#DATA_DICT#")
        #print(data_dict)
        print(data_dict.keys())
        print(data_dict["im_H"])
        print(data_dict["im_W"])
        
        
        out_dict = gdrn_predictor.inference(data_dict)
        
        print("\n#OUT_DICT#")
        #print(out_dict)
        print(out_dict.keys())
        
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        
        print("\n#POSES#")
        #print(poses)
        print(poses.keys())
        
        gdrn_predictor.gdrn_visualization_own(batch=data_dict, out_dict=out_dict, image=rgb_img)#, frame_count=cnt)

def test2():
    # prepare ground truth data
    #gt_data_path = "datasets/test/scene_gt.json"
    #gt_data = utils.read_json(gt_data_path)

    image_paths = get_image_list(osp.join(PROJ_ROOT,"datasets/test/rgb2"), osp.join(PROJ_ROOT,"datasets/test/depth2"))
    yolo_predictor = YoloPredictor(
                       exp_name="yolox-x",
                       config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
                       ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_ycb/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test/model_final.pth"),
                       fuse=True,
                       fp16=False
                     )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth"),
        #camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/itodd/camera.json"),
        path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
    )

    for cnt, (rgb_img, depth_img) in enumerate(image_paths):
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        outputs = yolo_predictor.inference(image=rgb_img)
        print('\n#YOLO OUTPUT#')
        for output in outputs:
            print(np.array(output.cpu()).shape)
            print(rgb_img.shape)
            yolo_predictor.visual_yolo(output,rgb_img,list(gdrn_predictor.objs.values()))

        
        data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=depth_img)
        
        print("\n#PARSED YOLO OUTPUT")
        print(data_dict)
        print(data_dict.keys())

        # sort out results under certain treshold
        #treshold = 0.8
        #ids_under_treshold = []
        #for count, value in enumerate(data_dict['score'].cpu()):
        #    if value < treshold:
        #        ids_under_treshold.append(count)
        #print(ids_under_treshold)
        #for key, tensor in data_dict.items():
        #    if tensor.size(0) > 0:
        #        arr = np.array(tensor.cpu())
        #        print(key)
        #        print(tensor)
        #        print(arr)
        #        arr_new = np.delete(arr,ids_under_treshold)
        #        new_tensor = torch.tensor(arr_new,device = tensor.device)
        #        data_dict[key] = new_tensor
        #        #data_dict[key] = torch.masked_select(tensor, torch.tensor([i not in ids_under_treshold for i in range(tensor.size(0))], device=tensor.device))
        #print(data_dict)

        out_dict = gdrn_predictor.inference(data_dict)
        print("\n#GDRN OUTPUT#")
        #print(out_dict)
        print(out_dict.keys())
        
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        
        print("\n#PARSED GDRN OUTPUT (POSES)#")
        #print(poses)
        print(poses.keys())
        for key,value in poses.items():
            print(str(key),'')
            print(value)
        
        gdrn_predictor.gdrn_visualization_own(batch=data_dict, out_dict=out_dict, image=rgb_img,gt_data=gt_data)#, frame_count=cnt)

def test3():
    image_paths = get_image_list(osp.join(PROJ_ROOT,"datasets/test/rgb2"), osp.join(PROJ_ROOT,"datasets/test/depth2"))
    yolo_predictor = YoloPredictor(
                       exp_name="yolox-x",
                       config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
                       ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_ycb/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test/model_final.pth"),
                       fuse=True,
                       fp16=False
                     )
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_final_wo_optim.pth"),
        #camera_json_path=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/itodd/camera.json"),
        path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
    )

    for cnt, (rgb_img, depth_img) in enumerate(image_paths):
        rgb_img = cv2.imread(rgb_img)
        if depth_img is not None:
            depth_img = cv2.imread(depth_img, 0)
        outputs = yolo_predictor.inference(image=rgb_img)
        #for output in outputs:
        #	print(output.cpu())
        #	print(rgb_img.shape)
        #	yolo_predictor.visual_yolo(output,rgb_img,["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28"])
        #print("#OUTPUTS#")
        #print(outputs)
        
        data_dict = gdrn_predictor.preprocessing(outputs=outputs, image=rgb_img, depth_img=depth_img)
        
        print("\n#DATA_DICT#")
        #print(data_dict)
        print(data_dict.keys())
        
        
        out_dict = gdrn_predictor.inference(data_dict)
        
        print("\n#OUT_DICT#")
        #print(out_dict)
        print(out_dict.keys())
        
        poses = gdrn_predictor.postprocessing(data_dict, out_dict)
        
        print("\n#POSES#")
        #print(poses)
        print(poses.keys())
        
        gdrn_predictor.gdrn_visualization_own(batch=data_dict, out_dict=out_dict, image=rgb_img)#, frame_count=cnt)

if __name__ == "__main__":
    test2()
