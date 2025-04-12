
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import numpy as np

class IndicSceneTextDataset(Dataset):
    def __init__(self, root_dir, languages, split='train', transform=None, task='detection'):
        """
        Args:
            root_dir: Root directory of the dataset
            languages: List of languages to include
            split: 'train', 'test', or 'val'
            transform: Optional transforms to apply
            task: 'detection', 'language_id', 'joint', or 'end_to_end'
        """
        self.root_dir = root_dir
        self.languages = languages
        self.split = split
        self.transform = transform
        self.task = task
        self.samples = self._load_samples()
        
        # Map language names to IDs for classification
        self.language_to_id = {
            'bengali': 0,
            'english': 1,
            'gujarati': 2,
            'hindi': 3,
            'kannada': 4,
            'malayalam': 5,
            'marathi': 6,
            'oriya': 7,
            'punjabi': 8,
            'tamil': 9,
            'telugu': 10,
            'symbol': 11
        }

    def _load_samples(self):
        samples = []
        for lang in self.languages:
            lang_dir = os.path.join(self.root_dir, lang, self.split)
            images_dir = os.path.join(lang_dir, 'images')
            annotations_dir = os.path.join(lang_dir, 'annotations')
            
            # Skip if directory doesn't exist
            if not os.path.exists(images_dir):
                print(f"Directory not found: {images_dir}")
                continue
                
            # Read image paths
            with open(os.path.join(lang_dir, 'images.txt'), 'r') as f:
                image_files = f.read().splitlines()
            
            # Read corresponding labels
            with open(os.path.join(lang_dir, 'labels.txt'), 'r') as f:
                labels = f.read().splitlines()
            
            for img_file, text_label in zip(image_files, labels):
                img_path = os.path.join(images_dir, os.path.basename(img_file))
                
                # Get annotation file path
                annotation_file = os.path.join(annotations_dir, 
                                              os.path.splitext(os.path.basename(img_file))[0] + '.json')
                
                # Use default bounding box if annotation doesn't exist
                if not os.path.exists(annotation_file):
                    # Create a default annotation covering the whole image
                    img = Image.open(img_path)
                    width, height = img.size
                    bbox = [[0, 0], [width, 0], [width, height], [0, height]]
                    annotations = [{
                        'points': bbox,
                        'text': text_label,
                        'language': lang
                    }]
                else:
                    # Load annotations from file
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                
                samples.append({
                    'image_path': img_path,
                    'annotations': annotations,
                    'language': lang
                })
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Get annotations
        annotations = sample['annotations']
        
        # Prepare output based on task
        if self.task == 'detection':
            # For detection task, return image and bounding boxes
            boxes = [ann['points'] for ann in annotations]
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'image_path': sample['image_path']
            }
            
        elif self.task == 'language_id':
            # For language identification task, return cropped image and language
            if len(annotations) > 0:
                # Use the first annotation
                ann = annotations[0]
                points = np.array(ann['points'])
                
                # Get bounding box
                x_min = points[:, 0].min()
                y_min = points[:, 1].min()
                x_max = points[:, 0].max()
                y_max = points[:, 1].max()
                
                # Crop image to bounding box
                cropped_img = image.crop((x_min, y_min, x_max, y_max))
                
                if self.transform:
                    cropped_img = self.transform(cropped_img)
                
                # Get language ID
                lang_id = self.language_to_id.get(sample['language'], 0)
                
                return {
                    'image': cropped_img,
                    'language_id': torch.tensor(lang_id, dtype=torch.long),
                    'image_path': sample['image_path']
                }
            else:
                # If no annotation, use the whole image
                if self.transform:
                    image = self.transform(image)
                
                lang_id = self.language_to_id.get(sample['language'], 0)
                
                return {
                    'image': image,
                    'language_id': torch.tensor(lang_id, dtype=torch.long),
                    'image_path': sample['image_path']
                }
                
        elif self.task == 'joint':
            # For joint detection and language ID task
            boxes = []
            lang_ids = []
            
            for ann in annotations:
                boxes.append(ann['points'])
                lang_id = self.language_to_id.get(ann.get('language', sample['language']), 0)
                lang_ids.append(lang_id)
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'language_ids': torch.tensor(lang_ids, dtype=torch.long),
                'image_path': sample['image_path']
            }
            
        elif self.task == 'end_to_end':
            # For end-to-end detection and recognition
            boxes = []
            texts = []
            
            for ann in annotations:
                boxes.append(ann['points'])
                texts.append(ann['text'])
            
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'texts': texts,
                'image_path': sample['image_path']
            }
