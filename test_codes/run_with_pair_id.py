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
SAVE_DIR = 'output/pred_files/without_poses/0927'

parser = argparse.ArgumentParser()
parser.add_argument( "--pair_id", type=str, help='ID of image pairs')
args = parser.parse_args()

with open(IMAGE_PAIRS_PATH, 'r') as json_file:
    print('### LOADING IMAGE PAIRS ###')
    image_pairs = json.load(json_file)
    print('### FINISHED LOADING IMAGE PAIRS ###')

# loading model and checkpoint
params = get_default_dataset_config("plane_params")
ckpt = "./models/planeformers_eccv.pt"
mv_inference = MultiViewInference(params, ckpt)

print('### BEGIN INFERENCE ###')

pair = image_pairs[args.pair_id]
scene_id = pair['0']['scene_id']

img_list = []
rot_list = []
trans_list = [] 

for k, img in pair.items():
    img_list.append(join(RGB_IMAGE_DIR, scene_id, img['image_name']))

# making predictions
preds = mv_inference.inference(img_list)

viz_info = {}
viz_info['preds'] = preds
viz_info['imgs'] = img_list

# saving predictions
with open(join(SAVE_DIR, 'pair_' + args.pair_id + '.pkl'), 'wb') as f:
    pickle.dump(viz_info, f)
