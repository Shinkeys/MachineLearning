import os
import glob
from   PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.optim.lr_scheduler import CosineAnnealingLR
from   torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from   sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ImageDataset(Dataset):    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ResNet18Classifier(nn.Module):
    '''Transfer learning model with resnet18 backbone'''    
    def __init__(self, num_classes=1, freeze_backbone=False):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze feature extractor
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)

class ImageClassifier:
    """Main classifier class for training and inference"""
    
    def __init__(self, data_dir='Resources', img_size=128):
        self.data_dir = data_dir
        self.img_size = img_size
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: CUDA (NVIDIA GPU)")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using device: MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: CPU")
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def build_model(self, freeze_backbone=False):
        """
        Args:
            freeze_backbone: If True, freeze ResNet backbone and only train the classifier head.
                             Good for small datasets or quick training. Default False (fine-tune all)
        """
        self.model = ResNet18Classifier(num_classes=1, freeze_backbone=freeze_backbone).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel Architecture: Pretrained ResNet-18")
        print(f"  - ImageNet pretrained weights")
        print(f"  - Custom classifier head with dropout")
        print(f"  - Backbone frozen: {freeze_backbone}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
        
        return self.model
    
    def load_data(self, num_images=None, test_split=0.2, random_state=42):
        print(f"Loading images from {self.data_dir}...")
        
        # Get all image paths
        cat_images = sorted(glob.glob(os.path.join(self.data_dir, '*_cat.jpg')))
        dog_images = sorted(glob.glob(os.path.join(self.data_dir, '*_dog.jpg')))
        
        # Limit number of images if specified
        if num_images:
            images_per_class = num_images // 2
            cat_images = cat_images[:images_per_class]
            dog_images = dog_images[:images_per_class]
        
        # Combine paths and create labels (0 = cat, 1 = dog)
        all_images = cat_images + dog_images
        all_labels = [0] * len(cat_images) + [1] * len(dog_images)
        
        print(f"Total images loaded: {len(all_images)}")
        print(f"  - Cats: {len(cat_images)}")
        print(f"  - Dogs: {len(dog_images)}")
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            all_images, all_labels, 
            test_size=test_split, 
            random_state=random_state,
            stratify=all_labels
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        
        return X_train, X_val, y_train, y_val
    
    def create_dataloaders(self, X_train, X_val, y_train, y_val, batch_size=32):

        train_transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

        # Validation transforms - NO augmentation
        val_transform = transforms.Compose([
        transforms.Resize((self.img_size, self.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
        train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
        val_dataset   = ImageDataset(X_val, y_val, transform=val_transform)
        
        # pin_memory only works with CUDA, not MPS
        use_pin_memory = self.device.type == 'cuda'
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=use_pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
        
        print(f"Batch size: {batch_size}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=100, learning_rate=1e-4, 
              start_epoch=0, checkpoint_dir='checkpoints'):
        if self.model is None:
            raise ValueError("Model not built! Call build_model() first.")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        # BCE is fine here because the training data is cat or dogs(0 or 1) 
        # - binary data
        criterion = nn.BCELoss()
        
        start = 0
        if start_epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{start_epoch}.pth')
            if os.path.exists(checkpoint_path):
                print(f"\nLoading checkpoint from epoch {start_epoch}...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.history = checkpoint['history']
                start = start_epoch
                print(f"Resumed training from epoch {start_epoch}")
            else:
                print(f"\nWarning: Checkpoint for epoch {start_epoch} not found!")
                print(f"Starting training from scratch...\n")
        else:
            print("\nStarting training from scratch...\n")
        
        for epoch in range(start, epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                # Stats
                train_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            # Calculate training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predictions = (outputs >= 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            current_lr = optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_accuracy)
            self.history['lr'].append(current_lr)
            
            print(f"Epoch [{epoch+1}/{epochs}] LR: {current_lr:.6f} | "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': self.history,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': train_accuracy,
                    'val_acc': val_accuracy
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("\nTraining completed!")

    def predict(self, image_path):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            confidence = output.item()
        
        prediction = 'dog' if confidence >= 0.5 else 'cat'
        confidence = confidence if confidence >= 0.5 else (1 - confidence)
        
        return prediction, confidence
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        if 'lr' in self.history and len(self.history['lr']) > 0:
            ax3.plot(self.history['lr'], label='Learning Rate', color='green')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule (Cosine Annealing)')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Training history plot saved as 'training_history.png'")
        plt.show()
    
    def save_model(self, filepath='dog_cat_model.pth'):
        if self.model is None:
            raise ValueError("No model to save!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'img_size': self.img_size
        }, filepath)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    classifier = ImageClassifier(data_dir='Resources', img_size=224)
    
    X_train, X_val, y_train, y_val = classifier.load_data(
        num_images=12500,
        test_split=0.2
    )

    train_loader, val_loader = classifier.create_dataloaders(
        X_train, X_val, y_train, y_val,
        batch_size=32 # Adjust based on GPU memory
    )
    
    # freeze_backbone=False: Fine-tune entire network (recommended for 96%+ accuracy)
    # freeze_backbone=True: Only train classifier head (faster, but lower accuracy ceiling)
    classifier.build_model(freeze_backbone=False)
    
    classifier.train(
        train_loader, 
        val_loader,
        epochs=50, # weights are already trained, so less epochs are needed
        learning_rate=3e-5,
        start_epoch=50,
        checkpoint_dir='checkpoints'
    )
    
    classifier.save_model('dog_cat_model_final.pth')
    classifier.plot_history()
    

    # Make predictions on a test image
    # prediction, confidence = classifier.predict('path/to/test_image.jpg')
    # print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")