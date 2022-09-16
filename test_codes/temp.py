import quaternion
import numpy as np
import torch
import json
from pytorch3d import transforms

j = open('../matterport/dataset/mp3d_planercnn_json/image_pairs_info.json')
image_pairs = json.load(j)

img_1 = image_pairs['61']['0']
img_2 = image_pairs['61']['1']

print(img_1['scene_id'])
print(img_1['image_name'])
print(img_2['image_name'])

q1 = np.array(img_1['camera']['rotation'])
q2 = np.array(img_2['camera']['rotation'])
t1 = np.array(img_1['camera']['position'])
t2 = np.array(img_2['camera']['position'])

t1.resize((3, 1))
t2.resize((3, 1))

# q1 = quaternion.quaternion(q1[0], q1[1], q1[2], q1[3])
# q2 = quaternion.quaternion(q2[0], q2[1], q2[2], q2[3])

# r1 = quaternion.as_rotation_matrix(q1)
# r2 = quaternion.as_rotation_matrix(q2)

q1 = torch.Tensor(q1)
q2 = torch.Tensor(q2)

r1 = transforms.quaternion_to_matrix(q1)
r2 = transforms.quaternion_to_matrix(q2)

r1 = r1 / np.linalg.norm(r1)
r2 = r2 / np.linalg.norm(r2)

p1 = np.append(r1, t1, axis=1)
p1 = np.append(p1, np.array([[0, 0, 0, 1]]), axis=0)

p2 = np.append(r2, t2, axis=1)
p2 = np.append(p2, np.array([[0, 0, 0, 1]]), axis=0)

print(p1)
print(p2)