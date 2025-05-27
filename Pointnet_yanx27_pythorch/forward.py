from test_semseg import *
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    NUM_CLASSES = 13
    area = "Area_3"
    room = "office_1"
    model_name = 'pointnet_sem_seg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES)
    checkpoint = torch.load('/home/joao/dev/LidarSegmentation/Pointnet_yanx27_pythorch/log/sem_seg/'+ model_name + '/checkpoints/best_model.pth', weights_only=False, map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    pc = np.zeros((1, 4096, 9))  # 1 sample, 4096 points, 9 features
    color = np.load('/home/joao/dev/datasets/s3dis/'+ area + '/' + room +'/color.npy')
    coord = np.load('/home/joao/dev/datasets/s3dis/'+ area + '/' + room +'/coord.npy')
    segment = np.load('/home/joao/dev/datasets/s3dis/'+ area + '/' + room +'/segment.npy')
    instance = np.load('/home/joao/dev/datasets/s3dis/'+ area + '/' + room +'/instance.npy')
    normal = np.load('/home/joao/dev/datasets/s3dis/'+ area + '/' + room +'/normal.npy')

    num_points = 4096
    random_indices = np.random.choice(color.shape[0], num_points, replace=False)
    color_sampled = color[random_indices]
    coord_sampled = coord[random_indices]
    print(f"color:{color.shape}")
    print(f"coord:{coord.shape}")
    print(f"color sampled:{color_sampled.shape}")
    print(f"coord sampled:{coord_sampled.shape}")


    color_normalized = color_sampled / 255.0
    coord_centered = coord_sampled.copy()
    coord_centered[:, 0] -= np.mean(coord_sampled[:, 0]) 
    coord_centered[:, 1] -= np.mean(coord_sampled[:, 1]) 
    #coord_centered[:, 2] -= np.mean(coord_sampled[:, 2]) 
    # Dont normalize the z!
    max_pt = np.max(coord_centered)
    coord_normalized = coord_centered / max_pt
    pc = np.concatenate((coord_centered, color_normalized, coord_normalized), axis=1)
    print(f"Max color {np.max(color_normalized)}")
    print(f"pc:{pc.shape}")
    pc_aux = pc.transpose(1,0)
    
    colors_segmentation = np.array([ 
        [255, 0, 0],       # 0 - ceiling      - red
        [0, 255, 0],       # 1 - floor        - green
        [0, 0, 255],       # 2 - wall         - blue
        [0, 0, 0],#[255, 255, 0],     # 3 - beam         - yellow
        [0, 0, 0],#[255, 0, 255],     # 4 - column       - magenta
        [0, 0, 0],#[0, 255, 255],     # 5 - window       - cyan
        [0, 0, 0],#[192, 192, 192],   # 6 - door         - light gray
        [0, 0, 0],#[128, 128, 128],   # 7 - table        - gray
        [0, 0, 0],#[128, 0, 0],       # 8 - chair        - maroon
        [0, 0, 0],#[128, 128, 0],     # 9 - sofa         - olive
        [0, 0, 0],#[0, 128, 0],       # 10 - bookcase    - dark green
        [0, 0, 0],#[128, 0, 128],     # 11 - board       - purple
        [0, 0, 0]          # 12 - clutter     - black
    ]) / 255.0


    pc_tensor = torch.tensor(pc_aux).float().unsqueeze(0)
    print(f"torch tensor shape: {pc_tensor.shape}")
    classifier = classifier.eval()
    with torch.no_grad():
        pred, _ = classifier(pc_tensor)
    print(f"pred shape: {pred.shape}")
    pred_labels = torch.argmax(pred, dim=2).squeeze(0)
    print(f"pred labels shape: {pred_labels.shape}")
    print(f"max label: {torch.max(pred_labels)}")

    # Create a grid of coordinates based on the downsampled array

    # Plotting the points

    #print(np.max(segment))
    
    point_cloud = o3d.geometry.PointCloud()
    pc_sampled_coords = pc[:, :3]
    print(f"coord shape: {coord.shape}")
    print(f"pc sampled coords shape: {pc_sampled_coords.shape}")
    print(f"colors shape: {colors_segmentation[0].shape}")
    #point_cloud.points = o3d.utility.Vector3dVector(coord)
    #point_colors = np.squeeze(np.array([colors_segmentation[label] for label in segment]), axis=1)
    point_cloud.points = o3d.utility.Vector3dVector(pc_sampled_coords)
    point_colors = np.array([colors_segmentation[label] for label in pred_labels])
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

    print(f"point colors shape: {point_colors.shape}")

    #point_cloud = point_cloud.voxel_down_sample(voxel_size=0.1)


    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
