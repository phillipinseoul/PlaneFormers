from planeformers.models.inference import *
from planeformers.figure_factory.visualize_mv import PlaneFormerInferenceVisualization
import pickle
import argparse
import json
import quaternion
from tqdm import tqdm
from os.path import join

IMAGE_PAIRS_PATH = '../matterport/dataset/mp3d_planercnn_json/image_pairs_info.json'
RGB_IMAGE_DIR = '../matterport/dataset/rgb'
SAVE_DIR = 'output/pred_files/without_poses'

with open(IMAGE_PAIRS_PATH, 'r') as json_file:
    print('### LOADING IMAGE PAIRS ###')
    image_pairs = json.load(json_file)
    print('### FINISHED LOADING IMAGE PAIRS ###')

# loading model and checkpoint
params = get_default_dataset_config("plane_params")
ckpt = "./models/planeformers_eccv.pt"
mv_inference = MultiViewInference(params, ckpt)

print('### BEGIN INFERENCE ###')
for idx, pairs in tqdm(image_pairs.items()):
    imgs = []
    poses = []
    scene_id = pairs['0']['scene_id']

    for img in pairs.values():
        imgs.append(join(RGB_IMAGE_DIR, scene_id, img['image_name']))

    # making predictions
    preds = mv_inference.inference(imgs)
    
    viz_info = {}
    viz_info['preds'] = preds
    viz_info['imgs'] = imgs

    # saving predictions
    with open(join(SAVE_DIR, 'pair_' + idx + '.pkl'), 'wb') as f:
        pickle.dump(viz_info, f)