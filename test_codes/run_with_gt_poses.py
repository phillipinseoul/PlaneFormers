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
SAVE_DIR = 'output/pred_files/with_poses'

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
        
        # make the camera pose matrix (4x4)
        rot_q = np.array(img['camera']['rotation'])                                # quaternion
        rot_q = quaternion.quaternion(rot_q[0], rot_q[1], rot_q[2], rot_q[3])
        rot_3x3 = quaternion.as_rotation_matrix(rot_q)                             # rotation: 3 x 3
        position = np.array(img['camera']['position'])
        position.resize((3, 1))                                                    # position

        pose = np.append(rot_3x3, position, axis=1)                                 
        pose = np.append(pose, np.array([[0, 0, 0, 1]]), axis=0)                   # camera matrix: 4 x 4
        poses.append(pose)

    # making predictions
    preds = mv_inference.inference_with_poses(imgs, poses)
    
    viz_info = {}
    viz_info['preds'] = preds
    viz_info['imgs'] = imgs

    # saving predictions
    with open(join(SAVE_DIR, 'pair_' + idx + '.pkl'), 'wb') as f:
        pickle.dump(viz_info, f)
