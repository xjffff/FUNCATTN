import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import wandb


def create_colormap(n_classes):
    """Create a distinct colormap for segmentation classes."""
    if n_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    
    colors = [cmap(i / n_classes) for i in range(n_classes)]
    return colors


def visualize_point_cloud_comparison(verts, gt_labels, pred_labels, 
                                     sample_name="Sample", n_classes=260,
                                     figsize=(20, 10), point_size=1):
    """
    Create a side-by-side comparison of ground truth vs predicted segmentation for point clouds.
    
    Args:
        verts: (N, 3) vertex positions
        gt_labels: (N,) ground truth labels
        pred_labels: (N,) predicted labels
        sample_name: name of the sample
        n_classes: number of segmentation classes
        figsize: figure size
        point_size: size of points in scatter plot
    
    Returns:
        matplotlib figure object, accuracy
    """
    # Convert to numpy if needed
    if torch.is_tensor(verts):
        verts = verts.cpu().numpy()
    if torch.is_tensor(gt_labels):
        gt_labels = gt_labels.cpu().numpy()
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.cpu().numpy()
    
    # Create colormap
    colormap = create_colormap(n_classes)
    
    # Get colors for each vertex based on labels
    gt_colors = np.array([colormap[label] for label in gt_labels])
    pred_colors = np.array([colormap[label] for label in pred_labels])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Ground truth visualization
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
                c=gt_colors, s=point_size, alpha=0.8)
    
    # Predicted visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
                c=pred_colors, s=point_size, alpha=0.8)
    
    # Set the aspect ratio and limits for both subplots
    max_range = np.array([verts[:, 0].max() - verts[:, 0].min(),
                          verts[:, 1].max() - verts[:, 1].min(),
                          verts[:, 2].max() - verts[:, 2].min()]).max() / 2.0
    
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(False)
        ax.set_facecolor('white')
    
    # Compute accuracy for this sample
    accuracy = (gt_labels == pred_labels).mean() * 100
    
    ax1.set_title(f'Ground Truth\n{sample_name}', fontsize=14, fontweight='bold')
    ax2.set_title(f'Predicted (Acc: {accuracy:.2f}%)\n{sample_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, accuracy


def visualize_point_cloud_error_map(verts, gt_labels, pred_labels,
                                    sample_name="Sample", figsize=(10, 10),
                                    point_size=1):
    """
    Visualize prediction errors on the point cloud.
    
    Args:
        verts: (N, 3) vertex positions
        gt_labels: (N,) ground truth labels
        pred_labels: (N,) predicted labels
        sample_name: name of the sample
        figsize: figure size
        point_size: size of points in scatter plot
    
    Returns:
        matplotlib figure object
    """
    # Convert to numpy if needed
    if torch.is_tensor(verts):
        verts = verts.cpu().numpy()
    if torch.is_tensor(gt_labels):
        gt_labels = gt_labels.cpu().numpy()
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.cpu().numpy()
    
    # Compute errors (0 = correct, 1 = incorrect)
    errors = (gt_labels != pred_labels).astype(float)
    
    # Create colors: green for correct, red for incorrect
    colors = np.zeros((len(errors), 4))
    colors[errors == 0] = [0.2, 0.8, 0.2, 0.9]  # Green for correct
    colors[errors == 1] = [0.9, 0.2, 0.2, 0.9]  # Red for incorrect
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
               c=colors, s=point_size, alpha=0.8)
    
    # Set the aspect ratio and limits
    max_range = np.array([verts[:, 0].max() - verts[:, 0].min(),
                          verts[:, 1].max() - verts[:, 1].min(),
                          verts[:, 2].max() - verts[:, 2].min()]).max() / 2.0
    
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=30, azim=45)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    accuracy = (1 - errors.mean()) * 100
    ax.set_title(f'Error Map - {sample_name}\n(Green=Correct, Red=Error, Acc={accuracy:.2f}%)',
                 fontsize=14, fontweight='bold')
    
    ax.grid(False)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    
    return fig


def log_point_cloud_segmentation_visualizations(args, model, test_loader, device, input_features,
                                                 n_samples=3, n_classes=260, epoch=0,
                                                 point_size=1):
    """
    Generate and log point cloud segmentation visualizations to wandb.
    
    Args:
        model: the segmentation model
        test_loader: test data loader
        device: torch device
        input_features: type of input features ('xyz' or 'hks')
        n_samples: number of samples to visualize
        n_classes: number of segmentation classes
        epoch: current epoch number
        point_size: size of points in scatter plot
    """
    import sys
    import os
    import src.models as models
    
    model.eval()
    
    # Separate lists for different visualization types
    comparison_images = []
    error_map_images = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= n_samples:
                break
            
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data
            
            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = models.geometry.compute_hks_autoscale(evals, evecs, 16)
            
            if args.model == 'pointnet2':
                xyz_input = verts.T.unsqueeze(0)  # (1, 3, N)
                pointnet_input = torch.cat([xyz_input, xyz_input, xyz_input], dim=1)  # (1, 9, N)
                preds, _ = model(pointnet_input)
                preds = preds.squeeze(0)  # (N, n_class)
            else:
                # Apply the model
                preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            
            # Get predicted labels
            pred_labels = torch.max(preds, dim=1).indices
            
            # Create comparison visualization
            fig_comparison, accuracy = visualize_point_cloud_comparison(
                verts, labels, pred_labels,
                sample_name=f"Test Sample {i+1}",
                n_classes=n_classes,
                figsize=(20, 10),
                point_size=point_size
            )
            
            # Create error map
            fig_error = visualize_point_cloud_error_map(
                verts, labels, pred_labels,
                sample_name=f"Test Sample {i+1}",
                figsize=(20, 10),
                point_size=point_size
            )
            
            # Add to separate lists
            comparison_images.append(
                wandb.Image(fig_comparison, caption=f"Sample {i+1} - Acc: {accuracy:.2f}%")
            )
            error_map_images.append(
                wandb.Image(fig_error, caption=f"Sample {i+1} - Error Map")
            )
            
            # Close figures to free memory
            plt.close(fig_comparison)
            plt.close(fig_error)
    
    # Log visualizations separately by type
    wandb.log({
        "visualizations/comparisons": comparison_images,
        "visualizations/error_maps": error_map_images,
    }, step=epoch)
    
    model.train()