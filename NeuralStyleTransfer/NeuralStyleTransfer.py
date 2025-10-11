import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class ImageLoader:
    def __init__(self, image_size=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.image_size = image_size
        self.device = device
        self.original_size = None
        
        # VGG19 expects images normalized with these values
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path, store_original_size=False):
        image = Image.open(image_path).convert('RGB')
        
        if store_original_size:
            self.original_size = image.size
        
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def tensor_to_image(self, tensor, target_size=None):
        image = tensor.clone().detach().cpu().squeeze(0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        
        # Clamp to valid range
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()
        
        # Upscale if target size is specified
        if target_size is not None:
            from PIL import Image as PILImage
            # Convert to PIL Image for resizing
            image_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            image_pil = image_pil.resize(target_size, PILImage.LANCZOS)
            image = np.array(image_pil) / 255.0
        
        return image


class VGG19FeatureExtractor(nn.Module):    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(VGG19FeatureExtractor, self).__init__()
        
        self.device = device

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.content_layers = ['21']  # conv4_2
        
        self.model = vgg.to(device).eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        style_features = []
        content_features = []
        
        # Extract features from the model
        for name, layer in self.model._modules.items():
            x = layer(x)
            
            if name in self.style_layers:
                style_features.append(x)
            
            if name in self.content_layers:
                content_features.append(x)
        
        return style_features, content_features


class NeuralStyleTransfer:    
    def __init__(self, content_path, style_path, image_size=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.image_loader = ImageLoader(image_size=image_size, device=self.device)
        
        self.content_image = self.image_loader.load_image(content_path, store_original_size=True)
        self.style_image = self.image_loader.load_image(style_path)
        
        # Store original images for display
        self.original_content_img = Image.open(content_path).convert('RGB')
        self.original_texture_img = Image.open(style_path).convert('RGB')
        
        # Store original size for upscaling the result
        self.original_size = self.image_loader.original_size
        print(f"Original input image size: {self.original_size}")
        print(f"Training image size: {image_size}x{image_size}")
        
        print(f"Content image shape: {self.content_image.shape}")
        print(f"Style image shape: {self.style_image.shape}")
        
        print("\nLoading VGG19 model...")
        self.feature_extractor = VGG19FeatureExtractor(device=self.device)
        
        print("Extracting features from images...\n")
        self.extract_features()
    
    def extract_features(self):
        with torch.no_grad():
            style_features, _ = self.feature_extractor(self.style_image)
            print(f"Extracted {len(style_features)} style feature maps from texture.jpg")
            for i, feat in enumerate(style_features):
                print(f"  Style layer {i+1}: shape {feat.shape}")
            
            _, content_features = self.feature_extractor(self.content_image)
            print(f"\nExtracted {len(content_features)} content feature map from input.jpg")
            for i, feat in enumerate(content_features):
                print(f"  Content layer {i+1}: shape {feat.shape}")
        
        self.style_features = style_features
        self.content_features = content_features
        
        # Initialize generated image (start from content image)
        self.generated_image = self.content_image.clone().requires_grad_(True)
    
    @staticmethod
    def gram_matrix(tensor):
        batch, channels, height, width = tensor.size()
        features = tensor.view(batch * channels, height * width)
        gram = torch.mm(features, features.t())
        # Normalize by number of elements
        return gram.div(batch * channels * height * width)
    
    def compute_content_loss(self, generated_features):
        loss = 0
        for gen_feat, target_feat in zip(generated_features, self.content_features):
            loss += nn.MSELoss()(gen_feat, target_feat)
        return loss
    
    def compute_style_loss(self, generated_features):
        loss = 0
        for gen_feat, target_feat in zip(generated_features, self.style_features):
            gen_gram = self.gram_matrix(gen_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += nn.MSELoss()(gen_gram, target_gram)
        return loss
    
    def train(self, num_iterations=1000, content_weight=1, style_weight=1e6, learning_rate=1e-2, show_every=100):
        """Train the model by optimizing the generated image"""
        print(f"\nStarting training for {num_iterations} iterations...")
        print(f"Content weight: {content_weight}")
        print(f"Style weight: {style_weight}")
        print(f"Learning rate: {learning_rate}\n")
        
        # Thanks to Adam optimizer as always
        optimizer = optim.Adam([self.generated_image], lr=learning_rate)
        
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        fig.canvas.manager.set_window_title('Neural Style Transfer - Training Progress')
        
        for iteration in range(1, num_iterations + 1):
            optimizer.zero_grad()
            
            # Extract features from current generated image
            gen_style_features, gen_content_features = self.feature_extractor(self.generated_image)
            
            # Compute losses
            c_loss = self.compute_content_loss(gen_content_features)
            s_loss = self.compute_style_loss(gen_style_features)
            
            # Total loss (weighted combination)
            total_loss = content_weight * c_loss + style_weight * s_loss
            
            # Backpropagate
            total_loss.backward()
            
            # Update generated image
            optimizer.step()
            
            # Clamp pixel values to valid range
            with torch.no_grad():
                self.generated_image.clamp_(-2.5, 2.5)
            
            if iteration % show_every == 0 or iteration == 1:
                print(f"Iteration {iteration}/{num_iterations}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Content Loss: {c_loss.item():.4f}")
                print(f"  Style Loss: {s_loss.item():.4f}\n")
                
                for ax in axes:
                    ax.clear()
                
                # Use original images directly, only upscale the result
                input_img = np.array(self.original_content_img) / 255.0
                texture_img = np.array(self.original_texture_img) / 255.0
                result_img = self.image_loader.tensor_to_image(self.generated_image, target_size=self.original_size)
                
                axes[0].imshow(input_img)
                axes[0].set_title('Input Image\n(Original)', fontsize=12, fontweight='bold', pad=10)
                axes[0].axis('off')
                
                axes[1].imshow(texture_img)
                axes[1].set_title('Texture Image\n(Original)', fontsize=12, fontweight='bold', pad=10)
                axes[1].axis('off')
                
                axes[2].imshow(result_img)
                axes[2].set_title(f'Result - Iteration {iteration}\n(Upscaled)', fontsize=12, fontweight='bold', pad=10)
                axes[2].axis('off')
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
        
        plt.ioff()
        print("Training completed!")


def main():
    content_path = Path("Resources/input.jpg")
    style_path = Path("Resources/texture1.jpg")
    
    nst = NeuralStyleTransfer(
        content_path=content_path,
        style_path=style_path,
        image_size=128
    )
    
    nst.train(
        num_iterations=1000,
        content_weight=1e1,
        style_weight=1e4,
        learning_rate=1e-1,
        show_every=100
    )

    # Visualize the result
    plt.show()


if __name__ == "__main__":
    main()