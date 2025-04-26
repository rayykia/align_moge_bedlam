from typing import List, Tuple
import utils3d
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import cv2
from BEDLAM.train.core.tester import Tester
from BEDLAM.train.utils.renderer_pyrd import Renderer
from BEDLAM.train.models.head.smplx_cam_head import SMPLXCamHead
from BEDLAM.train.core.config import update_hparams
from moge.utils.io import save_ply
from moge.model.v1 import MoGeModel
from PIL import Image
from transformers import SamProcessor, SamModel
from multi_person_tracker import MPT
from argparse import Namespace
from smplx import SMPLX
import trimesh
from loguru import logger
import warnings
warnings.filterwarnings("ignore")


logger.info("Using device: {}".format(torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"))


def run_detector(all_image_folder,):
    """Run the multi-person tracker to get bounding boxes."""
    mot = MPT(
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        batch_size=1,
        display=False,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=416,
    )
    bboxes = []
    for image_folder in all_image_folder:
        bboxes.append(mot.detect(image_folder))

    return bboxes


def run_sam(image, bounding_boxes):
    sam_checkpoint = "facebook/sam-vit-base"
    processor = SamProcessor.from_pretrained(sam_checkpoint)
    model = SamModel.from_pretrained(sam_checkpoint).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    inputs = processor(
        images=image,
        input_boxes=[bounding_boxes],
        # multimask_output=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        original_sizes=inputs["original_sizes"],
        reshaped_input_sizes=inputs["reshaped_input_sizes"]
    )
    masks = np.array([mask[0] for mask in masks[0].cpu().numpy()])
    del model

    return masks


def run_moge_maps(image, device):
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    moge_model.eval()

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(
        image / 255,
        dtype=torch.float32,
        device=device).permute(2, 0, 1)

    logger.info("Image shape: %s", image.shape)
    logger.info("Running MoGe model...")
    output = moge_model.infer(image)
    logger.info("MoGe output received.")

    point_map = output["points"].cpu().numpy()
    depth_map = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()
    inrinsics = output["intrinsics"].cpu().numpy()
    del moge_model

    return point_map, depth_map, mask, inrinsics


def visualize_moge_maps(moge_points: np.array,
                        moge_depth: np.array, moge_mask: np.array = None):
    """Visualize the MoGe model output.

    Args:
        moge_points (np.array): The point map from MoGe.
        moge_depth (np.array): The depth map from MoGe.
        moge_mask (np.array, optional): The mask from MoGe. Defaults to None.
    """
    plt.figure(figsize=(15, 6))
    plt.imshow(moge_depth)
    plt.colorbar(label='Depth')
    plt.title("Depth Map")
    plt.savefig('./moge_output/depth_map.png')
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.imshow(moge_points)
    plt.title("Point Map")
    plt.savefig('./moge_output/point_map.png')
    plt.close()

    if moge_mask is not None:
        plt.figure(figsize=(15, 6))
        plt.imshow(moge_mask)
        plt.title("MoGe Mask")
        plt.savefig('./moge_output/moge_mask.png')
        plt.close()


def visualize_mask(masks, env_depth):
    for i, mask in enumerate(masks):
        masked_img = np.zeros_like(env_depth)
        masked_img[mask != 0] = env_depth[mask != 0]
        plt.figure(figsize=(15, 6))
        plt.imshow(masked_img)
        plt.colorbar(label='Depth')
        plt.title(f"Depth Map {i}")
        plt.savefig(f'./moge_output/depth_map_{i}.png')
        plt.close()


def points2mesh(image_arr: np.array,
                point_map: np.array,
                depth_map: np.array,
                moge_mask: np.array) -> Tuple[np.array,
                                              np.array,
                                              np.array]:
    """Turn the point map into a mesh.
    The mesh is created by triangulating the point map and coloring it with 
    the image then saved to a PLY file.

    Args:
        image_arr (np.array): Input image.
        point_map (np.array): The point map from MoGe.
        depth_map (np.array): The depth map from MoGe.
        moge_mask (np.array): The mask from MoGe.
        align_scale (float, optional): _description_. Defaults to None.

    Returns:
        Return (Tuple[np.array, np.array, np.array]): The mesh representation 
        of the point map (vertices, faces, vertex_colors).
    """
    normals, normals_mask = utils3d.numpy.points_to_normals(
        point_map, moge_mask)

    height, width = image_arr.shape[:2]
    faces, vertices, vertex_colors, _ = utils3d.numpy.image_mesh(
        point_map,
        image_arr.astype(np.float32) / 255,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=moge_mask & ~(
            utils3d.numpy.depth_edge(
                depth_map,
                rtol=0.03,
                mask=moge_mask) & utils3d.numpy.normals_edge(
                normals,
                tol=5,
                mask=normals_mask)),
        tri=True
    )

    return vertices, faces, vertex_colors


def mesh2ply(filename, vertices, faces, vertex_colors):
    """Save the mesh to a PLY file.

    Parameters:
        vertices (numpy.array): The vertices of the mesh.
        faces (numpy.array): The faces of the mesh.
        vertex_colors (numpy.array): The colors of the vertices.
        filename (str): The filename to save the mesh.
    """
    save_ply(
        filename,
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )
    logger.info(f"Mesh saved to {filename}")


def masked_mean(depth_map, mask):
    """Calculate the mean of the depth map using the mask.

    Parameters:
        depth_map (numpy.array): The depth map from MoGe.
        mask (numpy.array): The mask from MoGe.

    Returns:
        mean_depth (float): The mean depth value.
    """
    masked_depth = depth_map[mask != 0]
    mean_depth = np.mean(masked_depth)
    return mean_depth


def run_bedlam(detection: List) -> dict:
    """Run BEDLAM to get SMPL-X parameters.

    Args:
        detection (List): bounding boxes of people.

    Returns:
        dict: SMPL-X parameters.
    """

    args = Namespace(
        cfg='BEDLAM/configs/demo_bedlam_cliff.yaml',
        ckpt='data/ckpt/bedlam_cliff.ckpt',
        image_folder='./data_drc',
        output_folder='./ply/bedlam_raw',
        tracker_batch_size=1,
        display=False,
        detector='yolo',
        yolo_img_size=416,
        eval_dataset=None,
        dataframe_path='data/ssp_3d_test.npz',
        data_split='test'
    )

    tester = Tester(args)

    logger.info("Running BEDLAM on images...")
    hmr_output = tester.infer_smplx([args.image_folder], detection)
    logger.info("BEDLAM output received.")
    del tester.model

    return hmr_output[0]


def smplx2mesh(hmr_output: dict, save=False) -> List:
    """Convert SMPL-X parameters to mesh.

    Args:
        hmr_output (dict): SMPL-X parameters.
        save (bool, optional): set if save ply files. Defaults to True.

    Returns:
        List: List of meshes.
    """
    smplx_path = "data/body_models/smplx/models/smplx/SMPLX_NEUTRAL.npz"
    model = SMPLX(model_path=smplx_path, gender='neutral', batch_size=1)
    faces = model.faces

    all_verts = hmr_output['vertices'].cpu().numpy()  # shape (10475, 3)
    all_trans = hmr_output['pred_cam_t'].cpu().numpy()

    meshes = []
    for i in range(all_verts.shape[0]):
        verts_translated = all_verts[i] + all_trans[i]
        mesh = trimesh.Trimesh(
            vertices=verts_translated,
            faces=faces,
            process=False)
        meshes.append(mesh)
        if save:
            mesh.export('./ply/bedlam_raw/human_{}.ply'.format(i))

    return meshes


def project_vertices(vertices, focal_length, image_size,  pred_cam_t = None):
    # N, V, _ = vertices.shape
    try:
        f_x, f_y = focal_length
    except:
        f_x = f_y = focal_length
    img_H, img_W = image_size
    
    # apply translation
    if pred_cam_t is not None:
        verts_trans = vertices + pred_cam_t.unsqueeze(1)
    else:
        verts_trans = vertices
    
    # perspective division
    verts_proj = verts_trans[..., :2] / np.clip(verts_trans[..., 2:], a_min=1e-5, a_max=None)
    
    # print(f'{f_x.shape =}')
    # print(f'{img_W.shape = }')
    verts_proj[..., 0] = verts_proj[..., 0] * f_x + img_W/ 2.0
    verts_proj[..., 1] = verts_proj[..., 1] * f_y + img_H / 2.0

    return verts_proj


def scale_factor(depth_mean: np.array,
                 hmr_meshes: List[trimesh.Trimesh],
                 focal_length: float,
                 imgsz: Tuple,
                 masks: np.array) -> float:
    """Calculate the scale factor to align the meshes with the depth map.

    Args:
        depth_mean (np.array): The mean depth values from MoGe.
        hmr_meshes (List): List of meshes from BEDLAM.

    Returns:
        float: The scale factor.
    """
    z_unit = np.array([0, 0, 1])

    mesh_ref_depth = np.zeros_like(depth_mean)
    for i, (mesh, mask) in enumerate(zip(hmr_meshes, masks)):
        dot_product = np.dot(mesh.face_normals, z_unit)
        front_faces = np.where(dot_product < 0)[0]

        front_vertices_idx = np.unique(mesh.faces[front_faces])

        front_vertices = mesh.vertices[front_vertices_idx]

        projected_vertices  =project_vertices(
            front_vertices, focal_length, imgsz
        )
        u_raw = projected_vertices[:, 0]
        v_raw = projected_vertices[:, 1]
        H, W = mask.shape
        inside_image_mask = (u_raw >= 0) & (u_raw < W) & (v_raw >= 0) & (v_raw < H)
        u = u_raw.round().astype(int).clip(0, W-1)
        v = v_raw.round().astype(int).clip(0, H-1)
        mask_values = mask[v, u]
        final_mask = inside_image_mask & (mask_values > 0)

        visiable_veritices = front_vertices[final_mask]


        # x = visiable_veritices[:, 0]
        # y = visiable_veritices[:, 1]
        # plt.figure(figsize=(8, 8))
        # # plt.imshow(mask)
        # plt.scatter(x, y, s=0.5)  # s is marker size
        # plt.gca().invert_yaxis()  # Important: y-axis down like image coordinates
        # plt.axis('equal')  # Keep aspect ratio
        # plt.show()


        vertices_z = visiable_veritices[:, -1]
        front_depth_mean = np.mean(vertices_z)
        mesh_ref_depth[i] = front_depth_mean

    scale_factor = mesh_ref_depth / depth_mean
    print("Mesh scale alignment ratio: ", scale_factor)
    scale = np.nanmean(scale_factor)
    logger.info(f"Scale factor: {scale}")
    if np.isnan(scale):
        logger.warning("Scale factor is NaN. Using default scale of 1.")
        scale = 1
    return scale


def scale_env(hmr_output, focal_length, img, mask):
    img_h = torch.tensor(height).cuda().float()
    img_w = torch.tensor(width).cuda().float()
    reprojected_vert = project_vertices(
        hmr_smplx['vertices'], hmr_smplx['pred_cam_t'], (focal_length[0], focal_length[0]), (img_h[0], img_w[0])
    )
    pass
    

if __name__ == '__main__':

    input_dir = ['./data_drc']
    img = Image.open('./data_drc/big_bang.jpg').convert("RGB")
    img_arr = np.array(img)
    height, width = img_arr.shape[:2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detection = run_detector(input_dir)
    detection_arr = np.array(detection).squeeze()
    bboxes = [[x - w / 2, y - h / 2, x + w / 2, y + h / 2]
              for x, y, w, h in detection_arr]

    masks = run_sam(img, bboxes)

    env_pts, env_depth, moge_mask, intrinsics = run_moge_maps(img, device)

    visualize = False
    if visualize:
        visualize_moge_maps(env_pts, env_depth)
        visualize_mask(masks, env_depth)

    vertices, faces, vertex_colors = points2mesh(
        img_arr, env_pts, env_depth, moge_mask)
    # mesh2ply('./ply/env/env.ply', vertices, faces, vertex_colors)

    depth_mean = []
    for i, mask in enumerate(masks):
        depth = masked_mean(env_depth, mask)
        depth_mean.append(depth)
        logger.info(f"Mean depth of object {i}: {depth:.2f}")
    depth_mean = np.array(depth_mean)

    hmr_smplx = run_bedlam(detection)

    for key, tensor in hmr_smplx.items():
        print(f"{key}: {tensor.shape}")

    img_h = torch.tensor(height).repeat(5).cuda().float()
    img_w = torch.tensor(width).repeat(5).cuda().float()
    focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
    pred_vertices_array = (
        (hmr_smplx['vertices'] + hmr_smplx['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
    )
    smplx_cam_head = SMPLXCamHead(img_res = 224).to(device)
    renderer = Renderer(
        focal_length = focal_length[0],
        img_w = img_w[0],
        img_h = img_h[0],
        faces = smplx_cam_head.smplx.faces,
        same_mesh_color=False
    )
    front_view = renderer.render_front_view(
        pred_vertices_array
    )

    # moge intrinsics estimation
    f_x = (intrinsics[0, 0] * img_w[0]).cpu().numpy()
    f_y = (intrinsics[1, 1] * img_h[0]).cpu().numpy()


    hmr_meshes = smplx2mesh(hmr_smplx, save=True)
    logger.info("BEDLAM SMPL-X meshes generated.")

    focal_length = focal_length.cpu().numpy()
    img_h = img_h.cpu().numpy()
    img_w = img_w.cpu().numpy()

    # scale_fac = scale_factor(depth_mean, hmr_meshes, focal_length[0], (img_h[0], img_w[0]), masks)
    scale_fac = scale_factor(depth_mean, hmr_meshes, (f_x, f_y), (img_h[0], img_w[0]), masks)

    mesh2ply(
        './ply/env/env_rescaled.ply',
        vertices * scale_fac,
        faces,
        vertex_colors)
    logger.info("Scaled alignment completed. PLY file saved.")
