from planeformers.utils.misc import *
import numpy as np
import torch
from pytorch3d import transforms

class MultiViewInference:

    def __init__(self, params, transformer_ckpt, device='cuda'):
        
        # sparseplane config and model
        cfg = get_cfg()
        get_sparseplane_cfg_defaults(cfg)
        cfg.merge_from_file(params.sparseplane_config)
        self.plane_rcnn_model = PlaneRCNN_Branch(cfg)

        # loading planeformer model
        self.transformer_model = PlaneFormer(params)
        self.transformer_model = torch.nn.DataParallel(self.transformer_model)
        self.transformer_model.to(device)
        ckpt = torch.load(transformer_ckpt)
        self.transformer_model.load_state_dict(ckpt['model_state_dict'])
        self.transformer_model.eval()

        # storing params
        self.params = params

        # k-means models from sparseplanes
        self.kmeans_rot = pickle.load(open(self.params.kmeans_rot, "rb"))
        self.kmeans_trans = pickle.load(open(self.params.kmeans_trans, "rb"))

        self.device = device

    @torch.no_grad()
    def predict_features(self, img_paths):

        num_images = len(img_paths)
        output_preds = {}

        tran_logits = torch.zeros(num_images * num_images, 32)
        rot_logits = torch.zeros(num_images * num_images, 32)

        for j in range(num_images):
            for k in range(num_images):
                
                output = self.plane_rcnn_model.inference(img_paths[j], img_paths[k])
                pred = self.plane_rcnn_model.process(output)

                if k == 0:
                    output_preds[str(j)] = pred['0']
                    output_preds[str(j)]['embedding'] = output['0']['instances'].embedding.cpu()
                    output_preds[str(j)]['pred_plane'] = pred['0']['pred_plane'].cpu()
                    output_preds[str(j)]['pred_depth'] = pred['0']['pred_depth'].cpu()
                    output_preds[str(j)]['embeddingbox'] = {'pred_boxes': pred['corrs']['0']['embeddingbox']['pred_boxes'].tensor.cpu(), \
                        'scores': pred['corrs']['0']['embeddingbox']['scores'].cpu()}

                tran_logits[j * num_images + k, :] = pred['camera']['logits']['tran'].cpu().reshape(-1)
                rot_logits[j * num_images + k, :] = pred['camera']['logits']['rot'].cpu().reshape(-1)

        output_preds['pred_camera'] = {}
        output_preds['pred_camera']['tran_logits'] =  tran_logits
        output_preds['pred_camera']['rot_logits'] = rot_logits
        output_preds['num_images'] = num_images

        return output_preds

    @torch.no_grad()
    def build_connectivity_graph(self, features):

        num_images = features['num_images']
        conn_matrix = torch.zeros(num_images, num_images)

        # building connectivity matrix
        for i in range(num_images):
            for j in range(num_images):

                emb_i = torch.squeeze(features[str(i)]['embedding'], dim=0)
                emb_j = torch.squeeze(features[str(j)]['embedding'], dim=0)

                dist_mat = torch.cdist(emb_i, emb_j)

                row_min, _ = torch.min(dist_mat, dim=0)
                col_min, _ = torch.min(dist_mat, dim=1)
                sigma = torch.median(dist_mat.reshape(-1))
                
                conn_matrix[i, j] = torch.sum(torch.exp(-torch.pow(row_min, 2)/sigma**2)) + \
                    torch.sum(torch.exp(-torch.pow(col_min, 2)/sigma**2))


        # setting diagonal to -1 
        conn_matrix[torch.arange(num_images), torch.arange(num_images)] = -1
        mst = self.find_spanning_tree(conn_matrix) 
        mst = np.array(mst)

        return mst

    # buildings spanning tree with max weights and min edges (no cycles)
    @torch.no_grad()
    def find_spanning_tree(self, conn_mat):

        nodes = [0]
        edges = []
        conn_mat[:, 0] = -1

        tot_nodes = conn_mat.shape[0]

        while(len(nodes) < tot_nodes):

            # adding max weight connection
            max_row_vals, max_row_idx = torch.max(conn_mat[nodes, :], dim=1)
            sel_idx = torch.argmax(max_row_vals)

            node_i = nodes[sel_idx]
            node_j = max_row_idx[sel_idx]

            assert node_i in nodes and node_j not in nodes
            
            conn_mat[node_i, node_j] = -1
            conn_mat[node_j, node_i] = -1
            conn_mat[:, node_j] = -1

            edges.append([node_i, node_j])
            nodes.append(node_j)

        assert len(edges) == (tot_nodes - 1)

        return edges


    # processing feature of given view for transformer model
    # camera_info contains keys
    #   'rotation': quaternion object
    #   'position': (1, 3) np array
    @torch.no_grad()
    def process_features(self, view_features, camera_info):

        planes_global_params = torch.tensor(get_plane_params_in_global(view_features['pred_plane'].numpy().copy(), camera_info)).to(torch.float) * self.params.plane_param_scaling
        concat_emb = [view_features['embedding'].clone()]

        if hasattr(self.params, 'use_plane_params'):
            if self.params.use_plane_params:
                concat_emb.append(planes_global_params)
        else:
            concat_emb.append(planes_global_params)
        

        if self.params.use_plane_mask:
            masks_decoded = np.array([mask_util.decode(view_features['instances'][mask_idx]['segmentation']) for mask_idx in range(len(view_features['instances']))])
            warped_mask = warp_mask(masks_decoded, view_features['pred_plane'].numpy().copy(), camera_info)
            seg_masks = torch.flatten(
                torchvision.transforms.Resize((self.params.mask_height, self.params.mask_width))(torch.FloatTensor(warped_mask)),
                start_dim=1,
            )
            concat_emb.extend([seg_masks])

        embs = torch.cat(concat_emb, dim=1)

        view_ft = {}
        view_ft['emb'] = embs
        view_ft['num_planes'] = embs.shape[0]

        return view_ft


    @torch.no_grad()
    def run_model_pairwise(self, num_images, pose_paths, edges, features, device='cuda'):
        edge_cameras = []
        edge_preds = []
        
        for i in range(len(edges)):

            edge = edges[i, :]
            # TODO: suppose the ground truth camera poses are given, instead of predicting them
            # This is implemented only for TWO images.
            views = []

            target_pose = np.loadtxt(pose_paths[0])
            ref_pose = np.loadtxt(pose_paths[1])

            rel_pose = get_relative_camera_pose(
                torch.Tensor(target_pose), 
                torch.Tensor(ref_pose)
            )

            for j in range(num_images):
                camera_info_view = {}
                
                if j == 0:
                    rot_quat = rel_pose['rotation']
                    camera_info_view['rotation'] = quaternion.quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
                    camera_info_view['position'] = np.asarray(rel_pose['position']).reshape((1, 3))
                else: 
                    camera_info_view['rotation'] = quaternion.quaternion(1, 0, 0, 0)
                    camera_info_view['position'] = np.zeros((1, 3))

                view_features = self.process_features(features[str(edge[j])], camera_info_view)
                views.append(view_features)

            model_input = {}
            model_input['emb'] = torch.cat([
                views[0]['emb'],
                views[1]['emb']
            ], dim=0).unsqueeze(0).to(torch.float).to(device)

            model_input['num_planes'] = torch.tensor([
                views[0]['emb'].shape[0],
                views[1]['emb'].shape[0]
            ]).unsqueeze(0).to(torch.long).to(device)

            model_preds = self.transformer_model(model_input, None)
            best_pred = model_preds

            # ensuring camera rot is a unit quaternion
            rot_quat /= np.linalg.norm(rot_quat)

            updated_camera = {}
            updated_camera['rotation'] = quaternion.quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
            updated_camera['position'] = np.asarray(rel_pose['position'])

            edge_cameras.append(updated_camera)
            edge_preds.append(best_pred)

        return edge_cameras, edge_preds


    # breadth first search on adjacency matrix
    @torch.no_grad()
    def bfs(self, adj_mat, init_vertex, dest_vertex):
        q = []
        visited = np.zeros((adj_mat.shape[0]), dtype=np.bool)

        q.append(init_vertex)
        visited[init_vertex] = True

        paths = [[init_vertex]]

        while len(q) > 0:
            cur_vert = q.pop(0)
            cur_path = paths.pop(0)

            neighbours = np.nonzero(adj_mat[cur_vert, :].reshape((-1,)))[0]

            for i in range(len(neighbours)):
                vert = neighbours[i]

                if not visited[vert]:
                    visited[vert] = True
                    q.append(vert)

                    tmp_path = copy.deepcopy(cur_path)
                    tmp_path.append(vert)

                    paths.append(tmp_path)

                    if vert == dest_vertex:
                        return paths[-1]


    @torch.no_grad()
    def lookup_edge_idx(self, i, j , edges):
        for e in range(edges.shape[0]):
            if i == edges[e, 0] and j == edges[e, 1]:
                return (e, False)
            elif j == edges[e, 0] and i == edges[e, 1]:
                return (e, True)
        raise Exception("Edge not found")



    @torch.no_grad()
    def chain_cameras(self, num_images, edges, edge_cameras, adj_mat):

        chained_cameras = []
        # chaining cameras (assuming view 0 is reference view)
        chained_cameras = [{'rotation': quaternion.quaternion(1, 0, 0, 0), 'position': np.zeros((1, 3))}]

        for i in range(1, num_images):
            cam = edge_cameras[0]

            init_rot = np.eye(3)
            init_trans = np.zeros((3, 1))

            rot = cam['rotation']
            trans = cam['position'].reshape((3, 1))

            init_rot = quaternion.as_rotation_matrix(rot)@init_rot
            init_trans = quaternion.as_rotation_matrix(rot)@init_trans + trans
            # init_rot = quaternion.as_rotation_matrix(rot).T@init_rot
            # init_trans = (quaternion.as_rotation_matrix(rot).T@(init_trans - trans))

            chained_camera_dict = {}
            chained_camera_dict['rotation'] = quaternion.from_rotation_matrix(init_rot)
            chained_camera_dict['position'] = init_trans.reshape((1, 3))

            chained_cameras.append(chained_camera_dict)

        return chained_cameras


    @torch.no_grad()
    def chain_cameras_wrapper(self, num_images, edges, edge_cameras, features, device='cuda'):

        # undirected graph adjacency mat
        adj_mat = np.zeros((num_images, num_images))
        adj_mat[edges[:, 0], edges[:, 1]] = 1
        adj_mat[edges[:, 1], edges[:, 0]] = 1

        # chaining cameras assuming view 0 is reference view
        chained_cameras = self.chain_cameras(num_images, edges, edge_cameras, adj_mat)

        return chained_cameras

    
    def merge_planes(self, num_images, edges, plane_corr_threshold, edge_preds, features):

        plane_list = []
        for i in range(edges.shape[0]):
            v0 = edges[i, 0]
            v1 = edges[i, 1]

            num_p0 = features[str(edges[i, 0])]['embedding'].shape[0]
            num_p1 = features[str(edges[i, 1])]['embedding'].shape[0]
            corr_mat =  edge_preds[i]['plane_corr'].squeeze(0)[:num_p0, num_p0: num_p0 + num_p1]

            cost_mat = 1 - corr_mat
            pred_corr = apply_km(cost_mat.cpu().numpy(), threshold=plane_corr_threshold)

            corrs = np.nonzero(pred_corr)
            plane_list.append(corrs)

        merged_planes = []
        for edge_num in range(edges.shape[0]):

            corrs = plane_list[edge_num]

            for i in range(len(corrs[0])):
                found = False
                for j in range(len(merged_planes)):
                    if merged_planes[j][edges[edge_num, 0]] == corrs[0][i]:
                        merged_planes[j][edges[edge_num, 1]] = corrs[1][i]
                        found = True
                        break
                    if merged_planes[j][edges[edge_num, 1]] == corrs[1][i]:
                        merged_planes[j][edges[edge_num, 0]] = corrs[0][i]
                        found = True
                        break
                
                if not found:
                    tmp = -np.ones((num_images,), dtype=np.intc)
                    tmp[edges[edge_num, 0]] = corrs[0][i]
                    tmp[edges[edge_num, 1]] = corrs[1][i]
                    merged_planes.append(tmp)

        return merged_planes, plane_list



    # list of img paths
    @torch.no_grad()
    def inference(self, img_paths, pose_paths):

        num_images = len(img_paths)
        
        # step 1: extracing features from images and building min spanning tree
        features = self.predict_features(img_paths)
        edges = self.build_connectivity_graph(features)

        # step 2: running model on each edge of graph to get camera guess
        # TODO: pass the path for GT camera poses to run_model_pairwise()
        edge_cameras, edge_preds = self.run_model_pairwise(num_images, pose_paths, edges, features, device=self.device)

        # step 3: chaining cameras
        chained_cameras = self.chain_cameras_wrapper(num_images, edges, edge_cameras, features, device=self.device)

        # step 4: merging planes
        merged_planes, _ = self.merge_planes(num_images, edges, self.params.plane_corr_threshold, edge_preds, features)

        return features, chained_cameras, None, merged_planes


def get_relative_camera_pose(target_pose, ref_pose):
    '''
    inputs
        - target_pose: 4x4 camera matrix
        - ref_pose: 4x4 camera matrix
    output:
        - rel_pose: 4x4 camera matrix
    '''
    target_rot = target_pose[:3, :3]
    target_rot_quat = transforms.matrix_to_quaternion(target_rot)
    target_rot_quat = np.asarray(target_rot_quat)
    target_rot_quat = quaternion.quaternion(
        target_rot_quat[0], target_rot_quat[1], target_rot_quat[2], target_rot_quat[3]
    )
    target_trans = target_pose[:3, 3]

    ref_rot = ref_pose[:3, :3]
    ref_rot_quat = transforms.matrix_to_quaternion(ref_rot)
    ref_rot_quat = np.asarray(ref_rot_quat)
    ref_rot_quat = quaternion.quaternion(
        ref_rot_quat[0], ref_rot_quat[1], ref_rot_quat[2], ref_rot_quat[3]
    )
    ref_trans = ref_pose[:3, 3]

    rel_rot = (ref_rot_quat.inverse() * target_rot_quat)
    
    rel_trans = get_relative_T_in_cam2_ref(
        quaternion.as_rotation_matrix(ref_rot_quat.inverse()),
        # np.linalg.inv(ref_rot),
        target_trans,
        ref_trans
    )
    rel_rot = quaternion.as_float_array(rel_rot).tolist()
    rel_trans = rel_trans.tolist()

    rel_pose = {'position': rel_trans, 'rotation': rel_rot}

    return rel_pose
