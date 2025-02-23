import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def deepfool_attack(model, image, mask, num_classes=2, max_iter=50, overshoot=0.02, device='cuda'):
    """
    DeepFool attack implementation for segmentation models
    """
    model.eval()
    image = image.clone().detach().to(device)
    mask = mask.clone().detach().to(device)
    
    batch_size, _, H, W = image.shape
    pert_image = image.clone().detach().requires_grad_(True)
    
    # Initialize perturbation
    r_total = torch.zeros_like(image).to(device)
    loop_i = 0
    
    with torch.enable_grad():
        while loop_i < max_iter:
            pert_image.requires_grad = True
            output = model(pert_image)
            
            # Convert mask to class indices (0 or 1)
            target = (mask > 0.5).float()
            
            # Calculate current classification
            pred = (torch.sigmoid(output) > 0.5).float()
            correct = (pred == target).all()
            if correct:
                break
                
            # Compute gradients
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
            grad = torch.autograd.grad(loss, pert_image, retain_graph=False)[0]
            
            # Compute perturbation direction
            w = grad / (grad.norm() + 1e-8)
            r_i = (loss + 1e-4) * w
            
            # Accumulate perturbation
            r_total = (r_total + r_i).clamp(-overshoot, overshoot)
            pert_image = torch.clamp(image + r_total, 0, 1).detach()
            
            loop_i += 1
            
    return pert_image.detach(), r_total.detach()

def plot_and_save_results(image, pert_image, clean_pred, pert_pred, mask, save_dir, filename):
    """Plot and save attack results with proper perturbation scaling"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 10))
    
    # Convert tensors to numpy arrays
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    pert_image_np = pert_image.detach().cpu().permute(1, 2, 0).numpy()
    perturbation_np = pert_image_np - image_np  # Range [-0.02, 0.02]
    
    # Normalize perturbation for visualization
    max_val = np.abs(perturbation_np).max()
    perturbation_normalized = (perturbation_np + max_val) / (2 * max_val)  # [0, 1]
    
    # Get predictions
    clean_pred_np = clean_pred.detach().cpu().squeeze().numpy()
    pert_pred_np = pert_pred.detach().cpu().squeeze().numpy()
    mask_np = mask.detach().cpu().squeeze().numpy()

    # Plot images
    plt.subplot(2, 3, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    
    plt.subplot(2, 3, 2)
    plt.imshow(pert_image_np)
    plt.title('Perturbed Image')
    
    plt.subplot(2, 3, 3)
    plt.imshow(perturbation_normalized, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Perturbation (Normalized)')
    
    plt.subplot(2, 3, 4)
    plt.imshow(clean_pred_np, cmap='gray')
    plt.title('Clean Prediction')
    
    plt.subplot(2, 3, 5)
    plt.imshow(pert_pred_np, cmap='gray')
    plt.title('Perturbed Prediction')
    
    plt.subplot(2, 3, 6)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Ground Truth')
    
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def run_deepfool_attack(model, val_loader, save_dir='deepfool_attack_results', device='cuda'):
    """Run DeepFool attack on validation set"""
    model = model.to(device).eval()
    os.makedirs(save_dir, exist_ok=True)
    
    dice_scores = []
    num_samples = 0
    
    for batch_idx, (images, masks) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)
        
        # Generate adversarial examples
        pert_images, _ = deepfool_attack(model, images, masks, device=device)
        
        with torch.no_grad():
            # Get predictions
            clean_outputs = model(images)
            clean_preds = (torch.sigmoid(clean_outputs) > 0.5).float()
            
            pert_outputs = model(pert_images)
            pert_preds = (torch.sigmoid(pert_outputs) > 0.5).float()
            
            # Calculate Dice score
            dice = dice_score(pert_preds, masks)
            dice_scores.append(dice.item())
            
        # Save visualizations for first 5 samples
        for i in range(min(5, images.size(0))):
            plot_and_save_results(
                image=images[i],
                pert_image=pert_images[i],
                clean_pred=clean_preds[i],
                pert_pred=pert_preds[i],
                mask=masks[i],
                save_dir=save_dir,
                filename=f'batch_{batch_idx}_sample_{i}.png'
            )
            
        num_samples += images.size(0)
        
    # Calculate final metrics
    avg_dice = np.mean(dice_scores)
    print(f'\nDeepFool Attack Results:')
    print(f'Average Dice Score: {avg_dice:.4f}')
    
    return avg_dice

def dice_score(pred, target):
    smooth = 1e-5
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

