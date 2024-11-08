#!/usr/bin/env python3
# dlm_attack_refactored_lbfgs_fixed.py

import argparse
import os
import pickle
import logging
import copy  

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips

from PIL import Image

from fedml_api.standalone.superfednas.elastic_nn.ofa_resnets_32x32_10_26 import OFAResNets32x32_10_26

def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_weight_updates(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        model_paths = pickle.load(f)
    
    model_updates = []
    for round_num in sorted(model_paths.keys()):
        update_path = model_paths[round_num]
        model_updates.append((round_num, update_path))
    
    logging.info(f"Loaded weight updates for {len(model_updates)} rounds from '{pickle_path}'.")
    return model_updates

def load_ground_truth_images(ground_truth_dir, target_label=None):
    if not os.path.exists(ground_truth_dir):
        raise FileNotFoundError(f"Ground truth directory not found at {ground_truth_dir}")
    
    label_to_image = {}
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    for filename in os.listdir(ground_truth_dir):
        if not filename.endswith('.png'):
            continue 
    
        filepath = os.path.join(ground_truth_dir, filename)
        
        try:
            label_str = filename.split('_')[-1]  # e.g., 'label3.png'
            label = int(label_str.replace('label', '').replace('.png', ''))
        except (IndexError, ValueError) as e:
            logging.warning(f"Filename '{filename}' does not match the expected format. Skipping.")
            continue
        

        if target_label is not None and label != target_label:
            continue
        
        # If we've already loaded an image for this label, skip
        if label in label_to_image:
            continue
        
        # Load and transform the image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, H, W]
        
        label_to_image[label] = image_tensor
        logging.info(f"Loaded ground truth image '{filename}' with label {label}.")
        
        # If focusing on a specific label and we've loaded it, break
        if target_label is not None and label in label_to_image:
            break
    
    if not label_to_image:
        raise ValueError(f"No ground truth images found in '{ground_truth_dir}' for the specified criteria.")
    
    return label_to_image

def compute_accuracy(recovered_label, ground_truth_label):
    return int(recovered_label == ground_truth_label)

def compute_psnr_metric(recovered_image, ground_truth_image):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    psnr = compare_psnr(ground_truth_np, recovered_np, data_range=ground_truth_np.max() - ground_truth_np.min())
    return psnr

def compute_ssim_metric(recovered_image, ground_truth_image):
    recovered_np = recovered_image.squeeze().detach().cpu().numpy()
    ground_truth_np = ground_truth_image.squeeze().detach().cpu().numpy()
    
    # Debugging information
    logging.debug(f"Recovered image shape: {recovered_np.shape}")
    logging.debug(f"Ground truth image shape: {ground_truth_np.shape}")
    
    # Check if images have 3 channels; if so, transpose to [H, W, C]
    if recovered_np.ndim == 3 and recovered_np.shape[0] == 3:
        recovered_np = np.transpose(recovered_np, (1, 2, 0))  # [H, W, C]
    if ground_truth_np.ndim == 3 and ground_truth_np.shape[0] == 3:
        ground_truth_np = np.transpose(ground_truth_np, (1, 2, 0))  # [H, W, C]
    
    if recovered_np.ndim == 3:
        height, width, channels = recovered_np.shape
    elif recovered_np.ndim == 2:
        height, width = recovered_np.shape
        channels = None
    else:
        raise ValueError("recovered_np and ground_truth_np must be 2D or 3D arrays")
    
    # Determine appropriate win_size
    win_size = 7  # Default value
    if height < win_size or width < win_size:
        win_size = min(height, width)
        if win_size % 2 == 0:
            win_size -= 1  
        if win_size < 3:
            logging.error("Images are too small for SSIM computation with win_size < 3.")
            raise ValueError("Images are too small for SSIM computation.")
        logging.warning(f"Adjusted win_size to {win_size} due to small image dimensions.")
    
    try:
        ssim_score = compare_ssim(
            recovered_np,
            ground_truth_np,
            data_range=ground_truth_np.max() - ground_truth_np.min(),
            channel_axis=2 if channels == 3 else None,
            win_size=win_size
        )
    except ValueError as e:
        logging.error(f"SSIM computation failed: {e}")
        raise
    
    return ssim_score

def compute_lpips_score(recovered_image, ground_truth_image, loss_fn):
    try:
        device = next(loss_fn.parameters()).device
    except StopIteration:
        logging.error("LPIPS model has no parameters to determine device.")
        raise RuntimeError("LPIPS model has no parameters to determine device.")
    
    # Ensure the images are in [0,1] and moved to the LPIPS model's device
    recovered_image = torch.clamp(recovered_image, 0, 1).to(device)
    ground_truth_image = torch.clamp(ground_truth_image, 0, 1).to(device)
    
    # Normalize to [-1,1] as required by LPIPS
    recovered_lpips = (recovered_image * 2) - 1 
    ground_truth_lpips = (ground_truth_image * 2) - 1
    
    with torch.no_grad():
        lpips_score = loss_fn(recovered_lpips, ground_truth_lpips).item()
    
    return lpips_score

def find_closest_image(recovered_image, ground_truth_dict, metric_fn, same_class=False, target_label=None):
    best_score = -float('inf')  # Initialize to negative infinity for maximization metrics like PSNR
    closest_label = None
    closest_image = None
    
    for label, gt_image in ground_truth_dict.items():
        if same_class:
            if target_label is None:
                raise ValueError("target_label must be specified when same_class is True.")
            if label != target_label:
                continue  # Skip images not in the target class
        
        score = metric_fn(recovered_image, gt_image)
        
        if np.isnan(score):
            logging.warning(f"Metric function returned NaN for label {label}. Skipping.")
            continue
        
        if score > best_score:
            best_score = score
            closest_label = label
            closest_image = gt_image
    
    if closest_image is None:
        logging.warning(f"No closest image found for label {target_label}.")
    
    return closest_label, closest_image, best_score

def find_closest_image_same_class(recovered_image, ground_truth_dict, metric_fn, target_label):
    return find_closest_image(
        recovered_image,
        ground_truth_dict,
        metric_fn,
        same_class=True,
        target_label=target_label
    )

def visualize_recovered_data(recovered_image, ground_truth_image, closest_image_same_class, save_path=None):
    if recovered_image is None or ground_truth_image is None:
        logging.error("recovered_image or ground_truth_image is None.")
        raise ValueError("recovered_image and ground_truth_image must be valid tensors.")
    
    # Process Recovered Image
    recovered_np = recovered_image.detach().cpu().numpy()
    logging.debug(f"Original recovered_image shape: {recovered_np.shape}")
    
    # Remove batch dimension if present
    if recovered_np.ndim == 4 and recovered_np.shape[0] == 1:
        recovered_np = recovered_np.squeeze(0)
        logging.debug(f"Squeezed recovered_image shape: {recovered_np.shape}")
    elif recovered_np.ndim == 3 and recovered_np.shape[0] == 1:
        recovered_np = recovered_np.squeeze(0)
        logging.debug(f"Squeezed recovered_image shape: {recovered_np.shape}")
    
    # Check if channels are first and transpose if necessary
    if recovered_np.ndim == 3 and recovered_np.shape[0] == 3:
        recovered_np = np.transpose(recovered_np, (1, 2, 0))  # [H, W, C]
        logging.debug(f"Transposed recovered_image shape: {recovered_np.shape}")
    elif recovered_np.ndim == 2:
        # Single channel image, expand dims to [H, W, 1]
        recovered_np = np.expand_dims(recovered_np, axis=2)
        logging.debug(f"Expanded recovered_image shape: {recovered_np.shape}")
    
    # Validate shape after processing
    if recovered_np.ndim != 3 or recovered_np.shape[2] not in [1, 3]:
        logging.error(f"Recovered image has invalid shape after processing: {recovered_np.shape}")
        raise ValueError(f"Recovered image has invalid shape after processing: {recovered_np.shape}")
    
    # Clip values to [0,1]
    recovered_np = np.clip(recovered_np, 0, 1)
    
    # Process Ground Truth Image
    ground_truth_np = ground_truth_image.detach().cpu().numpy()
    logging.debug(f"Original ground_truth_image shape: {ground_truth_np.shape}")
    
    # Remove batch dimension if present
    if ground_truth_np.ndim == 4 and ground_truth_np.shape[0] == 1:
        ground_truth_np = ground_truth_np.squeeze(0)
        logging.debug(f"Squeezed ground_truth_image shape: {ground_truth_np.shape}")
    elif ground_truth_np.ndim == 3 and ground_truth_np.shape[0] == 1:
        ground_truth_np = ground_truth_np.squeeze(0)
        logging.debug(f"Squeezed ground_truth_image shape: {ground_truth_np.shape}")
    
    # Check if channels are first and transpose if necessary
    if ground_truth_np.ndim == 3 and ground_truth_np.shape[0] == 3:
        ground_truth_np = np.transpose(ground_truth_np, (1, 2, 0))  # [H, W, C]
        logging.debug(f"Transposed ground_truth_image shape: {ground_truth_np.shape}")
    elif ground_truth_np.ndim == 2:
        # Single channel image, expand dims to [H, W, 1]
        ground_truth_np = np.expand_dims(ground_truth_np, axis=2)
        logging.debug(f"Expanded ground_truth_image shape: {ground_truth_np.shape}")
    
    # Validate shape after processing
    if ground_truth_np.ndim != 3 or ground_truth_np.shape[2] not in [1, 3]:
        logging.error(f"Ground truth image has invalid shape after processing: {ground_truth_np.shape}")
        raise ValueError(f"Ground truth image has invalid shape after processing: {ground_truth_np.shape}")
    
    # Clip values to [0,1]
    ground_truth_np = np.clip(ground_truth_np, 0, 1)
    
    # Initialize Closest Same-Class Image
    closest_same_np = None
    if closest_image_same_class is not None:
        closest_same_np = closest_image_same_class.detach().cpu().numpy()
        logging.debug(f"Original closest_same_np shape: {closest_same_np.shape}")
        
        # Remove batch dimension if present
        if closest_same_np.ndim == 4 and closest_same_np.shape[0] == 1:
            closest_same_np = closest_same_np.squeeze(0)
            logging.debug(f"Squeezed closest_same_np shape: {closest_same_np.shape}")
        elif closest_same_np.ndim == 3 and closest_same_np.shape[0] == 1:
            closest_same_np = closest_same_np.squeeze(0)
            logging.debug(f"Squeezed closest_same_np shape: {closest_same_np.shape}")
        
        # Check if channels are first and transpose if necessary
        if closest_same_np.ndim == 3 and closest_same_np.shape[0] == 3:
            closest_same_np = np.transpose(closest_same_np, (1, 2, 0))  # [H, W, C]
            logging.debug(f"Transposed closest_same_np shape: {closest_same_np.shape}")
        elif closest_same_np.ndim == 2:
            # Single channel image, expand dims to [H, W, 1]
            closest_same_np = np.expand_dims(closest_same_np, axis=2)
            logging.debug(f"Expanded closest_same_np shape: {closest_same_np.shape}")
        
        # Validate shape after processing
        if closest_same_np.ndim != 3 or closest_same_np.shape[2] not in [1, 3]:
            logging.error(f"Closest same-class image has invalid shape after processing: {closest_same_np.shape}")
            closest_same_np = None  # Set to None to handle gracefully
        else:
            # Clip values to [0,1]
            closest_same_np = np.clip(closest_same_np, 0, 1)
    
    # Determine the number of columns based on the availability of closest_same_np
    num_cols = 3 if closest_same_np is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(10 * num_cols, 5))
    
    # Ground Truth Image
    axes[0].imshow(ground_truth_np.squeeze())
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Reconstructed Image
    axes[1].imshow(recovered_np.squeeze())
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # Closest Same-Class Image or Placeholder
    if closest_same_np is not None:
        axes[2].imshow(closest_same_np.squeeze())
        axes[2].set_title('Closest Same Class')
    else:
        # Display a placeholder image or a message
        if num_cols == 3:
            blank_image = np.ones_like(recovered_np)
            axes[2].imshow(blank_image)
            axes[2].set_title('No Closest Image Found')
        # If num_cols < 3, do nothing
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Visualization saved to '{save_path}'.")
        plt.close()
    else:
        plt.show()


def plot_metrics(loss_history, psnr_history, output_dir, round_num, target_label):
    plt.figure(figsize=(12, 5))

    # Loss History
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Total Loss')
    plt.title(f'Round {round_num} - Label {target_label} - Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # PSNR History
    plt.subplot(1, 2, 2)
    plt.plot(psnr_history, label='PSNR', color='orange')
    plt.title(f'Round {round_num} - Label {target_label} - PSNR History')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'metrics_round_{round_num}_label_{target_label}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"[*] Metrics plot saved to '{plot_path}'.")

def get_supernet():
    model = OFAResNets32x32_10_26().to(DEVICE)
    model.set_active_subnet(d=0, e=0.1, w=0)  
    model = model.get_active_subnet()
    for name, param in model.named_parameters():
        param.requires_grad = True
        if param.grad is not None:
            param.grad.zero_()
    
    return model

class DLM_Attack:

    def __init__(self, model, loss_fn, optimizer_class, optimizer_params, device, ground_truth_image, ground_truth_label):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.device = device
        self.reconstructed_data = torch.randn_like(ground_truth_image).float().to(self.device).requires_grad_(False)
        self.target_label = ground_truth_label
        self.ground_truth_data = ground_truth_image
        self.loss_history = []
        self.psnr_history = []
        self.history_images = [] 
        
    
    
    def reconstruct_input(self, target_grads, num_steps=1000):
        self.model.eval()
        self.reconstructed_data.requires_grad_(True)
        optimizer = self.optimizer_class([self.reconstructed_data], **self.optimizer_params)
        offset = 1e-8  # Small constant to prevent division by zero
    
        def closure():
            optimizer.zero_grad()
    
            # Forward pass with reconstructed data
            outputs = self.model(self.reconstructed_data)
            loss_ce = self.loss_fn(outputs, self.target_label.view(-1))
    
            # Compute gradients of loss_ce w.r.t. model parameters
            grads_current = torch.autograd.grad(loss_ce, self.model.parameters(), create_graph=True)
    
            # Compute gradient matching loss with normalized gradients
            grad_diff = 0.0
            for gx, gy in zip(grads_current, target_grads):
                gx_norm = gx.norm()
                gy_norm = gy.norm()
    
                gx_normalized = gx / (gx_norm + offset)
                gy_normalized = gy / (gy_norm + offset)
    
                # Sum of squared differences
                grad_diff += ((gx_normalized - gy_normalized) ** 2).sum()
    
            # Backward on grad_diff to update reconstructed data
            grad_diff.backward()
    
            # Record the loss
            self.loss_history.append(grad_diff.item())
    
            # Compute PSNR
            mse = torch.mean((self.reconstructed_data - self.ground_truth_data) ** 2).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            self.psnr_history.append(psnr)
    

            ss_dummy_data = self.reconstructed_data.clone().detach()
            ss_gt_data = self.ground_truth_data.clone().detach()
            LPIPS = dis_lpips(ss_dummy_data, ss_gt_data).item()
            logging.info(f"----------------------------------------------------")
            logging.info(f"Loss = {grad_diff.item():.6f}")
            logging.info(f"PSNR = {psnr:.2f} dB")
            logging.info(f"LPIPS = {LPIPS:.4f}")
            logging.info(f"----------------------------------------------------")
    
            return grad_diff
    

        for step in range(1, num_steps + 1):
            optimizer.step(closure)

        self.reconstructed_data.requires_grad_(False)
        self.reconstructed_data[self.reconstructed_data!=self.reconstructed_data]=0
        
        recovered_data = self.reconstructed_data.clone()

        with torch.no_grad():
            recovered_outputs = self.model(self.reconstructed_data)
            recovered_label = torch.argmax(recovered_outputs, dim=1).item()
    
        logging.info(f"[*] Reconstruction completed. Recovered Label: {recovered_label}")
    
        return recovered_data, recovered_label, self.loss_history, self.psnr_history

def apply_weight_update(model, weight_update):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weight_update:
                if param.shape != weight_update[name].shape:
                    logging.warning(f"Skipping parameter '{name}' due to shape mismatch: model has {param.shape}, weight update has {weight_update[name].shape}")
                    continue  
                param += weight_update[name].to(param.device)
            else:
                logging.warning(f"Parameter '{name}' not found in weight_update.")

def dlm_attack(client_pickle_path, server_pickle_path, ground_truth_dir, output_dir, num_steps=1000, lr=1.0, device=DEVICE):
    if not os.path.isfile(client_pickle_path):
        logging.error(f"Client pickle file not found at '{client_pickle_path}'.")
        return
    if not os.path.isfile(server_pickle_path):
        logging.error(f"Server pickle file not found at '{server_pickle_path}'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    

    # Load server and client weight updates
    try:
        server_weight_updates = load_weight_updates(server_pickle_path)
    except Exception as e:
        logging.error(f"Failed to load server weight updates: {e}")
        return

    try:
        client_weight_updates = load_weight_updates(client_pickle_path)
    except Exception as e:
        logging.error(f"Failed to load client weight updates: {e}")
        return

    if not client_weight_updates:
        logging.error("No client weight updates found. Exiting.")
        return

    # Initialize two models: one for c_(i-1) (server_prev) and one for c_i (client_current)
    model1 = get_supernet()
    model1.eval()
    logging.info("[*] Initialized model_server_prev (c_(i-1)).")

    model2 = get_supernet()
    model2.eval()
    logging.info("[*] Initialized model_client_current (c_i).")

    # Load ground truth images
    try:
        ground_truth_images_all = load_ground_truth_images(ground_truth_dir)  # Load one image per label
    except Exception as e:
        logging.error(f"Failed to load ground truth images: {e}")
        return

    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    global dis_lpips
    dis_lpips = lpips_fn

    target_label = 0  # Example: label 0
    if target_label not in ground_truth_images_all:
        logging.error(f"No ground truth image found for label {target_label} in '{ground_truth_dir}'. Exiting.")
        return

    ground_truth_data = ground_truth_images_all[target_label]
    ground_truth_label_tensor = torch.tensor([target_label], dtype=torch.long).to(device)

    # Initialize the DLM_Attack object with model_server_prev and LBFGS optimizer
    dlm_attacker = DLM_Attack(
        model=model1,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_class=optim.LBFGS,  
        optimizer_params={
            'lr': lr
        },
        device=device,
        ground_truth_image=ground_truth_data,
        ground_truth_label=ground_truth_label_tensor
    )
    previous_recovered_data = None
    num_iterations = 20
    for i in range(num_iterations):
        for i in range(2, len(client_weight_updates)):
            round_num_prev, update_path_prev = server_weight_updates[i-1]
            round_num_curr, update_path_curr = client_weight_updates[i]

            logging.info(f"[*] Processing Checkpoint Pair: Round {round_num_prev} -> Round {round_num_curr}")

            # Load and apply the previous checkpoint to model1
            if not os.path.exists(update_path_prev):
                logging.error(f"Weight update file not found at '{update_path_prev}'. Skipping Checkpoint Pair {round_num_prev} -> {round_num_curr}.")
                continue

            try:
                with open(update_path_prev, 'rb') as f:
                    weight_update_prev = pickle.load(f)
                apply_weight_update(model1, weight_update_prev)
                logging.info(f"Applied weight update for Round {round_num_prev} to model1.")
            except Exception as e:
                logging.error(f"Error loading/applying weight update for Round {round_num_prev}: {e}")
                continue

            # Load and apply the current checkpoint to model2
            if not os.path.exists(update_path_curr):
                logging.error(f"Weight update file not found at '{update_path_curr}'. Skipping Checkpoint Pair {round_num_prev} -> {round_num_curr}.")
                continue

            try:
                with open(update_path_curr, 'rb') as f:
                    weight_update_curr = pickle.load(f)
                apply_weight_update(model2, weight_update_curr)
                logging.info(f"Applied weight update for Round {round_num_curr} to model2.")
            except Exception as e:
                logging.error(f"Error loading/applying weight update for Round {round_num_curr}: {e}")
                continue

            # Compute target gradients: c_(i-1).param - c_i.param
            target_gradients = []
            for param1, param2 in zip(model1.parameters(), model2.parameters()):
                grad = param1.data - param2.data 
                target_gradients.append(grad.detach().clone())

            # Verify that target_gradients are non-zero
            if all(torch.all(g == 0) for g in target_gradients):
                logging.warning(f"All target gradients are zero for Checkpoint Pair {round_num_prev} -> {round_num_curr}. Skipping attack.")
                continue

            # Normalize target gradients
            #normalized_target_grads = [g / (g.norm() + 1e-8) for g in target_gradients]

            # Set model1 to model1's current state
            dlm_attacker.model = model1
            dlm_attacker.model.eval()

            # Perform the DLM attack to reconstruct the input using LBFGS
            try:
                recovered_data, recovered_label, loss_history, psnr_history = dlm_attacker.reconstruct_input(
                    target_grads=target_gradients,
                    num_steps=num_steps  
                )
            except Exception as e:
                logging.error(f"Error during reconstruction for Checkpoint Pair {round_num_prev} -> {round_num_curr}: {e}")
                continue

            # Compute Metrics
            try:
                accuracy = compute_accuracy(recovered_label, target_label)
                psnr = compute_psnr_metric(recovered_data, ground_truth_data)
                ssim_score = compute_ssim_metric(recovered_data, ground_truth_data)
                lpips_score = compute_lpips_score(recovered_data, ground_truth_data, lpips_fn)
            except Exception as e:
                logging.error(f"Error computing metrics for Checkpoint Pair {round_num_prev} -> {round_num_curr}: {e}")
                accuracy = 0
                psnr = float('nan')
                ssim_score = float('nan')
                lpips_score = float('nan')

            logging.info(f"Metrics for Checkpoint Pair {round_num_prev} -> {round_num_curr}, Label {target_label}:")
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"PSNR: {psnr:.2f} dB")
            logging.info(f"SSIM: {ssim_score:.4f}")
            logging.info(f"LPIPS: {lpips_score:.4f}")

            # Define unique output path for this checkpoint pair and label
            output_path = os.path.join(output_dir, f'recovered_round_{round_num_prev}_to_{round_num_curr}_label_{target_label}.pt')
            if i % 1 == 0:
                # Save Recovered Data and Metrics
                try:
                    torch.save({
                        'data': recovered_data,
                        'label': recovered_label,
                        'metrics': {
                            'Accuracy': accuracy,
                            'PSNR': psnr,
                            'SSIM': ssim_score,
                            'LPIPS': lpips_score
                        }
                    }, output_path)
                    logging.info(f"[*] Recovered data and metrics saved to '{output_path}'.")
                except Exception as e:
                    logging.error(f"Error saving recovered data for Checkpoint Pair {round_num_prev} -> {round_num_curr}: {e}")

            # Optionally, save the reconstructed image
            visual_path = os.path.join(output_dir, f'recovered_round_{round_num_prev}_to_{round_num_curr}_label_{target_label}.png')
            if i % 1 == 0:
                try:
                    # Find the closest same-class image for visualization
                    closest_label, closest_image, best_score = find_closest_image_same_class(
                        recovered_data,
                        ground_truth_images_all,
                        compute_psnr_metric,
                        target_label
                    )
                    visualize_recovered_data(
                        recovered_data,
                        ground_truth_data,
                        closest_image_same_class=closest_image,  # Pass the closest image
                        save_path=visual_path
                    )
                    logging.info(f"[*] Visualization saved to '{visual_path}'.")
                except Exception as e:
                    logging.error(f"Error during visualization for Checkpoint Pair {round_num_prev} -> {round_num_curr}: {e}")

                # Plot and save metrics
                try:
                    plot_metrics(loss_history, psnr_history, output_dir, f"{round_num_prev}_to_{round_num_curr}", target_label)
                except Exception as e:
                    logging.error(f"Error plotting metrics for Checkpoint Pair {round_num_prev} -> {round_num_curr}: {e}")

        logging.info("[*] DLM attack completed successfully.")

def main():
    parser = argparse.ArgumentParser(description='DLM Attack on SuperFedNAS with CIFAR-10 using LBFGS')
    
    parser.add_argument('--client_pickle', type=str, default='./fed_avg/client_weight_updates/all_client_model_updates.pkl',
                        help='Path to weight updates pickle file')
    parser.add_argument('--server_pickle', type=str, default='./fed_avg/server_model_updates/all_server_model_updates.pkl',
                        help='Path to server modeul updates pickle file')
    parser.add_argument('--ground_truth_dir', type=str, default='./ground_truths/',
                        help='Directory containing ground truth image files')
    parser.add_argument('--output_dir', type=str, default='./recovered_data/',
                        help='Directory to save recovered data and metrics')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=1,  
                        help='Learning rate for optimizer')
    parser.add_argument('--gpu_id', type=int, default=7,
                        help='GPU ID to use (0-7)')
    
    args = parser.parse_args()
    
    global DEVICE
    DEVICE = set_device(args.gpu_id)
    logging.info(f"Using device: {DEVICE} (GPU ID: {args.gpu_id})")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    dlm_attack(
        client_pickle_path=args.client_pickle,
        server_pickle_path=args.server_pickle,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,           
        num_steps=args.steps,
        lr=args.lr,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
