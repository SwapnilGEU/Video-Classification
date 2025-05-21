import os
import random
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pytorchvideo.models.hub import slow_r50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt

# UCF101 Dataset
class UCF101Dataset(Dataset):
    def __init__(self, csv_file, video_base_path, selected_labels=None, 
                 num_frames_per_video=16, resize=(112, 112), transform=None):
        self.video_base_path = video_base_path
        self.num_frames_per_video = num_frames_per_video
        self.resize = resize
        self.df = pd.read_csv(csv_file)
        if selected_labels is not None:
            self.df = self.df[self.df['label'].isin(selected_labels)].reset_index(drop=True)
        self.transform = transform
        self.valid_indices = list(range(len(self.df)))
        self.problematic_videos = set()
        unique_labels = sorted(self.df['label'].unique())
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.valid_indices)
    
    def extract_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"Error: Video {video_path} has no frames")
                cap.release()
                return None
            if total_frames < self.num_frames_per_video:
                print(f"Warning: Video {video_path} has only {total_frames} frames, using uniform sampling with repetition")
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)
            frames = []
            max_retries = 3
            for idx in frame_indices:
                success = False
                retry_count = 0
                while not success and retry_count < max_retries:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        success = True
                        break
                    retry_count += 1
                if not success:
                    for offset in range(1, min(10, total_frames)):
                        if idx - offset >= 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx - offset)
                            ret, frame = cap.read()
                            if ret:
                                success = True
                                break
                        if idx + offset < total_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx + offset)
                            ret, frame = cap.read()
                            if ret:
                                success = True
                                break
                if not success:
                    print(f"Warning: Could not read frame {idx} from {video_path}, using black frame")
                    frame = np.zeros((self.resize[0], self.resize[1], 3), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.resize)
                frames.append(frame)
            cap.release()
            if len(frames) != self.num_frames_per_video:
                print(f"Error: Expected {self.num_frames_per_video} frames but got {len(frames)} from {video_path}")
                return None
            frames_np = np.stack(frames).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
            if self.transform:
                frames_tensor = self.transform(frames_tensor)
            return frames_tensor
        except Exception as e:
            print(f"Exception processing video {video_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        try:
            valid_idx = self.valid_indices[idx]
            row = self.df.iloc[valid_idx]
            clip_rel_path = row['clip_path'].lstrip("/")
            video_path = os.path.join(self.video_base_path, clip_rel_path)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            frames = self.extract_frames(video_path)
            if frames is None:
                self.problematic_videos.add(valid_idx)
                remaining_videos = set(self.valid_indices) - self.problematic_videos
                if not remaining_videos:
                    print("Warning: All videos marked as problematic. Resetting problematic videos list.")
                    self.problematic_videos = set()
                    remaining_videos = set(self.valid_indices)
                new_idx = random.choice(list(remaining_videos))
                new_pos = self.valid_indices.index(new_idx)
                return self.__getitem__(new_pos)
            label = self.label_to_index[row['label']]
            return frames, label
        except Exception as e:
            print(f"Error in __getitem__ with idx {idx}: {e}")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

# Inverse normalization for visualization
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

# Check data loader
def check_dataloader(loader, label_to_index, num_samples=2):
    index_to_label = {v: k for k, v in label_to_index.items()}
    dataiter = iter(loader)
    sample_count = 0
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        try:
            frames, labels = next(dataiter)
        except StopIteration:
            print(f"Only {i} samples available in the loader")
            break
        video = frames[0]
        label_idx = labels[0].item()
        label_name = index_to_label[label_idx]
        video_denorm = torch.stack([denormalize(frame) for frame in video])
        num_frames_to_show = min(8, video.shape[0])
        frame_indices = np.linspace(0, video.shape[0] - 1, num_frames_to_show, dtype=int)
        for j, frame_idx in enumerate(frame_indices):
            frame = video_denorm[frame_idx].permute(1, 2, 0).cpu().numpy()
            plt.subplot(num_samples, num_frames_to_show, i * num_frames_to_show + j + 1)
            plt.imshow(frame)
            plt.axis('off')
            if j == 0:
                plt.title(f"Label: {label_name}")
        sample_count += 1
    plt.tight_layout()
    plt.suptitle("Sample Frames from Dataset", fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.savefig('dataloader_samples.png')
    plt.close()
    print(f"Successfully visualized {sample_count} samples from the data loader")
    print(f"Frames shape: {frames.shape}, Labels shape: {labels.shape}")

# Check batch processing
def check_batch_processing(loader, label_to_index, max_batches=5):
    print("\nChecking batch processing...")
    batch_count = 0
    class_counts = {label: 0 for label in label_to_index.keys()}
    index_to_label = {v: k for k, v in label_to_index.items()}
    for i, (frames, labels) in enumerate(loader):
        if i >= max_batches:
            break
        batch_count += 1
        print(f"Batch {i+1} - Frames tensor shape: {frames.shape}, Labels tensor shape: {labels.shape}")
        for label_idx in labels:
            label_name = index_to_label[label_idx.item()]
            class_counts[label_name] += 1
    print(f"\nProcessed {batch_count} batches successfully")
    print("Class distribution in processed batches:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} instances")

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device="cuda",
                save_path="best_model.pth", last_epoch_path="last_epoch.pth", scheduler=None,
                early_stopping_patience=15, plateau_factor=0.1, plateau_patience=5):
    model = model.to(device)
    start_time = datetime.now()
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    best_val_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    if scheduler is None:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=plateau_factor, patience=plateau_patience, verbose=True)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, unit="batch")
        batch_start_time = time.time()
        for inputs, labels in train_pbar:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1, 3, 4)
                print(f"Input shape after permute: {inputs.shape}")  # Debug print
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                if time.time() - batch_start_time > 0:
                    batch_speed = 1.0 / (time.time() - batch_start_time)
                    train_pbar.set_postfix({"batch/s": f"{batch_speed:.2f}"})
                batch_start_time = time.time()
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        epoch_train_loss = train_loss / len(train_loader.dataset) if train_total > 0 else float('inf')
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0
        epoch_val_loss = val_loss / len(val_loader.dataset) if val_total > 0 else float('inf')
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        is_best = epoch_val_acc > best_val_acc
        best_saved_message = ""
        if is_best:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'train_loss': epoch_train_loss,
                'history': history,
            }, save_path)
            best_saved_message = f" ✅ Saved new best model at Epoch {epoch+1} with Val Acc: {epoch_val_acc:.4f}"
        else:
            epochs_no_improve += 1
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Acc: {epoch_val_acc:.4f}, "
              f"LR: {current_lr:.6f}{best_saved_message}")
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_val_acc)
        elif scheduler is not None:
            scheduler.step()
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
            break
    total_time = datetime.now() - start_time
    print(f"Training completed in {total_time}")
    print(f"Final model saved at: {last_epoch_path}")
    print(f"Best model saved at: {save_path} (Epoch {best_epoch+1} with Val Acc: {best_val_acc:.4f})")
    return model, history

# --- Training Setup ---
# Define transforms
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize datasets
selected_labels = ['ApplyEyeMakeup', 'Archery', 'Basketball', 'Billiards', 'BoxingPunchingBag']
train_dataset = UCF101Dataset(
    csv_file='train_updated.csv',  # Updated path
    video_base_path='data',  # Updated path
    selected_labels=selected_labels,
    transform=transform
)
val_dataset = UCF101Dataset(
    csv_file='val_updated.csv',  # Updated path
    video_base_path='data',  # Updated path
    selected_labels=selected_labels,
    transform=transform
)
label_to_index = train_dataset.label_to_index

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Check data loaders
print("Checking training data loader...")
check_dataloader(train_loader, label_to_index)
print("\nChecking validation data loader...")
check_dataloader(val_loader, label_to_index)
print(f"\nTraining dataset size: {len(train_dataset)} videos")
print(f"Validation dataset size: {len(val_dataset)} videos")
check_batch_processing(train_loader, label_to_index)

# Initialize model
num_classes = len(selected_labels)
model = slow_r50(pretrained=True)
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Train the model
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    device=device,
    save_path="best_model.pth",
    last_epoch_path="last_epoch.pth",
    scheduler=scheduler,
    early_stopping_patience=15,
    plateau_factor=0.1,
    plateau_patience=5
)

print("Training complete!")
