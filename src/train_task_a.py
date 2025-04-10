import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.dataset import IndicSceneTextDataset
from src.model.detection import TextDetectionModel
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter

def collate_fn(batch):
    """Custom collate function for detection task."""
    images = [item['image'] for item in batch]
    targets = []
    
    for i, item in enumerate(batch):
        # Convert boxes to the format expected by the model
        boxes = item['boxes']
        if len(boxes) == 0:  # Skip images without annotations
            continue
            
        # Convert polygon points to bounding boxes (xmin, ymin, xmax, ymax)
        bbox_list = []
        for box in boxes:
            points = box.numpy()
            x_min = points[:, 0].min()
            y_min = points[:, 1].min()
            x_max = points[:, 0].max()
            y_max = points[:, 1].max()
            bbox_list.append([x_min, y_min, x_max, y_max])
            
        if len(bbox_list) == 0:
            continue
            
        target = {
            'boxes': torch.tensor(bbox_list, dtype=torch.float32),
            'labels': torch.ones((len(bbox_list),), dtype=torch.int64),  # All boxes are text
            'image_id': torch.tensor([i]),
            'area': (boxes[:, 0].max(dim=1)[0] - boxes[:, 0].min(dim=1)[0]) * 
                   (boxes[:, 1].max(dim=1)[0] - boxes[:, 1].min(dim=1)[0]),
            'iscrowd': torch.zeros((len(bbox_list),), dtype=torch.int64)
        }
        targets.append(target)
    
    return images, targets

def calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Calculate mean Average Precision for object detection.
    
    Args:
        pred_boxes: List of predicted bounding boxes
        pred_scores: List of confidence scores
        pred_labels: List of predicted labels
        gt_boxes: List of ground truth bounding boxes
        gt_labels: List of ground truth labels
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        mAP value
    """
    if len(gt_boxes) == 0:
        return 0.0
        
    if len(pred_boxes) == 0:
        return 0.0
    
    # Calculate IoU between each pred box and each gt box
    ious = torch.zeros(len(pred_boxes), len(gt_boxes))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = calculate_iou(pred_box, gt_box)
    
    # Sort predictions by confidence score
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    # Initialize variables for precision and recall calculation
    tp = torch.zeros(len(sorted_indices))
    fp = torch.zeros(len(sorted_indices))
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    # Calculate TP and FP
    for i, idx in enumerate(sorted_indices):
        # Find the best matching ground truth box
        max_iou, max_idx = torch.max(ious[idx], dim=0)
        
        if max_iou >= iou_threshold and not gt_matched[max_idx] and pred_labels[idx] == gt_labels[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Add start point (0,1) for AUC calculation
    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])
    
    # Calculate AP using the Area Under Curve (AUC) of precision-recall curve
    ap = torch.trapz(precisions, recalls)
    
    return ap.item()

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def calculate_fscore(precision, recall):
    """Calculate F-score from precision and recall."""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """Train the model for one epoch."""
    model.train()
    
    running_loss = 0.0
    epoch_loss = 0.0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Skip batch if there are no valid targets
        if len(targets) == 0:
            continue
            
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += losses.item()
        epoch_loss += losses.item()
        
        # Print statistics
        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / print_freq
            elapsed_time = time.time() - start_time
            logging.info(f"Epoch [{epoch}][{i+1}/{len(data_loader)}] Loss: {avg_loss:.4f} Time: {elapsed_time:.2f}s")
            running_loss = 0.0
            start_time = time.time()
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Skip batch if there are no valid targets
            if len(targets) == 0:
                continue
                
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Collect predictions and targets
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()
                
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # Filter predictions by confidence threshold
                keep = pred_scores > 0.5
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                
                all_predictions.append((pred_boxes, pred_scores, pred_labels))
                all_targets.append((gt_boxes, gt_labels))
    
    # Calculate metrics
    precisions = []
    recalls = []
    f_scores = []
    
    for (pred_boxes, pred_scores, pred_labels), (gt_boxes, gt_labels) in zip(all_predictions, all_targets):
        if len(gt_boxes) == 0:
            continue
            
        if len(pred_boxes) == 0:
            precisions.append(0)
            recalls.append(0)
            f_scores.append(0)
            continue
        
        # Calculate true positives and false positives
        tp = 0
        fp = 0
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        for i, pred_box in enumerate(pred_boxes):
            # Find the best matching ground truth box
            best_iou = 0
            best_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_iou >= 0.5:
                tp += 1
                gt_matched[best_idx] = True
            else:
                fp += 1
        
        # Calculate precision and recall
        precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
        f_score = calculate_fscore(precision, recall)
        
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
    
    # Calculate average metrics
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f_score = np.mean(f_scores) if f_scores else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f_score': avg_f_score
    }

def main(args):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = IndicSceneTextDataset(
        root_dir=args.data_root,
        languages=args.languages,
        split='train',
        transform=train_transform,
        task='detection'
    )
    
    val_dataset = IndicSceneTextDataset(
        root_dir=args.data_root,
        languages=args.languages,
        split='val',
        transform=val_transform,
        task='detection'
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = TextDetectionModel(num_classes=2)  # Background and text
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
    # Training loop
    best_f_score = 0.0
    for epoch in range(args.num_epochs):
        # Train for one epoch
        epoch_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq
        )
        
        # Evaluate on validation set
        metrics = evaluate(model, val_loader, device)
        
        # Log metrics
        logging.info(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, F-score: {metrics['f_score']:.4f}")
        
        # Update learning rate
        lr_scheduler.step(metrics['f_score'])
        
        # Save TensorBoard logs
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Metrics/precision', metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', metrics['recall'], epoch)
        writer.add_scalar('Metrics/f_score', metrics['f_score'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'metrics': metrics,
        }
        
        # Save latest checkpoint (overwrite previous)
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if metrics['f_score'] > best_f_score:
            best_f_score = metrics['f_score']
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            logging.info(f"New best model saved with F-score: {best_f_score:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    
    logging.info("Training completed!")
    logging.info(f"Best F-score: {best_f_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train text detection model')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='Dataset', help='Path to dataset root directory')
    parser.add_argument('--languages', type=str, nargs='+', default=['bengali', 'hindi', 'kannada', 'tamil', 'telugu'],
                        help='Languages to include in training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs/task_a',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    main(args)
