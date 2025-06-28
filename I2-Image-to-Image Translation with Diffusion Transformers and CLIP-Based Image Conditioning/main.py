import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, DDIMScheduler
from torchvision import transforms
import lpips
from pathlib import Path
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PositionalEmbedding(nn.Module):
   """Positional embeddings for patches"""


   def __init__(self, num_patches: int, hidden_size: int):
       super().__init__()
       self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02)


   def forward(self, x):
       return x + self.pos_embed




class TimestepEmbedder(nn.Module):
   """Embeds scalar timesteps into vector representations"""


   def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
       super().__init__()
       self.mlp = nn.Sequential(
           nn.Linear(frequency_embedding_size, hidden_size, bias=True),
           nn.SiLU(),
           nn.Linear(hidden_size, hidden_size, bias=True),
       )
       self.frequency_embedding_size = frequency_embedding_size


   @staticmethod
   def timestep_embedding(t, dim, max_period=10000):
       """Create sinusoidal timestep embeddings"""
       half = dim // 2
       freqs = torch.exp(
           -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
       ).to(device=t.device)
       args = t[:, None].float() * freqs[None]
       embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
       if dim % 2:
           embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
       return embedding


   def forward(self, t):
       t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
       t_emb = self.mlp(t_freq)
       return t_emb




class CLIPImageEncoder(nn.Module):
   """CLIP image encoder for conditioning"""


   def __init__(self, clip_model_name: str = "openai/clip-vit-large-patch14"):
       super().__init__()
       self.clip_model = CLIPModel.from_pretrained(clip_model_name)
       self.processor = CLIPProcessor.from_pretrained(clip_model_name)


       # Freeze CLIP parameters
       for param in self.clip_model.parameters():
           param.requires_grad = False


   def forward(self, images):
       """Extract CLIP embeddings from images"""
       with torch.no_grad():
           image_features = self.clip_model.get_image_features(images)
       return image_features




class AdaLNZero(nn.Module):
   """Adaptive Layer Normalization with Zero initialization"""


   def __init__(self, hidden_size: int, conditioning_size: int):
       super().__init__()
       self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.adaLN_modulation = nn.Sequential(
           nn.SiLU(),
           nn.Linear(conditioning_size, 6 * hidden_size, bias=True)
       )


       # Zero initialization
       nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
       nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


   def forward(self, x, c):
       """
       x: input tensor [B, N, D]
       c: conditioning tensor [B, D_cond]
       """
       shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
           self.adaLN_modulation(c).chunk(6, dim=1)


       # Reshape for broadcasting
       shift_msa = shift_msa.unsqueeze(1)
       scale_msa = scale_msa.unsqueeze(1)
       gate_msa = gate_msa.unsqueeze(1)
       shift_mlp = shift_mlp.unsqueeze(1)
       scale_mlp = scale_mlp.unsqueeze(1)
       gate_mlp = gate_mlp.unsqueeze(1)


       return (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)




class DiTBlock(nn.Module):
   """DiT transformer block with CLIP conditioning"""


   def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                conditioning_size: int = 768):
       super().__init__()
       self.hidden_size = hidden_size
       self.num_heads = num_heads
       head_dim = hidden_size // num_heads
       self.scale = head_dim ** -0.5


       # Layer norms
       self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)


       # Self-attention
       self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)


       # Cross-attention for CLIP conditioning
       self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
       self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)


       # MLP
       mlp_hidden_dim = int(hidden_size * mlp_ratio)
       self.mlp = nn.Sequential(
           nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
           nn.GELU(),
           nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
       )


       # AdaLN-Zero
       combined_conditioning_size = hidden_size + conditioning_size
       self.adaLN_modulation = AdaLNZero(hidden_size, combined_conditioning_size)


       # CLIP conditioning projection
       self.clip_proj = nn.Linear(conditioning_size, hidden_size)


   def forward(self, x, t_emb, clip_emb):
       """
       x: patch embeddings [B, N, D]
       t_emb: timestep embeddings [B, hidden_size]
       clip_emb: CLIP embeddings [B, conditioning_size]
       """
       # Combine timestep and CLIP embeddings for AdaLN
       combined_conditioning = torch.cat([t_emb, clip_emb], dim=1)


       # Get modulation parameters
       shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
           self.adaLN_modulation(x, combined_conditioning)


       # Self-attention with AdaLN
       norm_x = self.norm1(x)
       norm_x = norm_x * (1 + scale_msa) + shift_msa
       attn_out, _ = self.attn(norm_x, norm_x, norm_x)
       x = x + gate_msa * attn_out


       # Cross-attention with CLIP
       clip_proj = self.clip_proj(clip_emb).unsqueeze(1)  # [B, 1, D]
       norm_x_cross = self.norm_cross(x)
       cross_attn_out, _ = self.cross_attn(norm_x_cross, clip_proj, clip_proj)
       x = x + cross_attn_out


       # MLP with AdaLN
       norm_x = self.norm2(x)
       norm_x = norm_x * (1 + scale_mlp) + shift_mlp
       mlp_out = self.mlp(norm_x)
       x = x + gate_mlp * mlp_out


       return x




class PatchEmbed(nn.Module):
   """Convert latent to patch embeddings"""


   def __init__(self, latent_size: int = 32, patch_size: int = 2,
                in_channels: int = 4, embed_dim: int = 768):
       super().__init__()
       self.latent_size = latent_size
       self.patch_size = patch_size
       self.num_patches = (latent_size // patch_size) ** 2


       self.proj = nn.Conv2d(in_channels, embed_dim,
                             kernel_size=patch_size, stride=patch_size)


   def forward(self, x):
       B, C, H, W = x.shape
       x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
       x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
       return x




class CLIPConditionedDiT(nn.Module):
   """CLIP-conditioned Diffusion Transformer"""


   def __init__(
           self,
           latent_size: int = 32,
           patch_size: int = 2,
           in_channels: int = 4,
           hidden_size: int = 768,
           depth: int = 12,
           num_heads: int = 12,
           mlp_ratio: float = 4.0,
           clip_conditioning_size: int = 768,
   ):
       super().__init__()
       self.latent_size = latent_size
       self.patch_size = patch_size
       self.in_channels = in_channels
       self.out_channels = in_channels
       self.num_heads = num_heads
       self.hidden_size = hidden_size


       # Patch embedding
       self.patch_embed = PatchEmbed(latent_size, patch_size, in_channels, hidden_size)
       num_patches = self.patch_embed.num_patches


       # Positional embedding
       self.pos_embed = PositionalEmbedding(num_patches, hidden_size)


       # Timestep embedding
       self.timestep_embed = TimestepEmbedder(hidden_size)


       # CLIP encoder
       self.clip_encoder = CLIPImageEncoder()


       # DiT blocks
       self.blocks = nn.ModuleList([
           DiTBlock(hidden_size, num_heads, mlp_ratio, clip_conditioning_size)
           for _ in range(depth)
       ])


       # Final layer norm
       self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)


       # Output projection
       self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels)


       # Final AdaLN
       combined_conditioning_size = hidden_size + clip_conditioning_size
       self.final_adaLN = AdaLNZero(hidden_size, combined_conditioning_size)


       self.initialize_weights()


   def initialize_weights(self):
       """Initialize weights"""
       # Initialize patch embedding like nn.Linear
       w = self.patch_embed.proj.weight.data
       nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
       nn.init.constant_(self.patch_embed.proj.bias, 0)


       # Initialize timestep embedding MLP
       nn.init.normal_(self.timestep_embed.mlp[0].weight, std=0.02)
       nn.init.normal_(self.timestep_embed.mlp[2].weight, std=0.02)


       # Zero-out output layers
       nn.init.constant_(self.final_proj.weight, 0)
       nn.init.constant_(self.final_proj.bias, 0)


   def unpatchify(self, x):
       """Convert patches back to latent"""
       B = x.shape[0]
       H = W = int((x.shape[1]) ** 0.5)
       x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.out_channels)
       x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
       x = x.reshape(B, self.out_channels, H * self.patch_size, W * self.patch_size)
       return x


   def forward(self, x, t, clip_images):
       """
       x: noisy latents [B, C, H, W]
       t: timesteps [B]
       clip_images: conditioning images [B, 3, 224, 224]
       """
       # Get CLIP embeddings
       clip_emb = self.clip_encoder(clip_images)  # [B, 768]


       # Get timestep embeddings
       t_emb = self.timestep_embed(t)  # [B, hidden_size]


       # Patch embedding
       x = self.patch_embed(x)  # [B, num_patches, hidden_size]


       # Add positional embeddings
       x = self.pos_embed(x)


       # Apply DiT blocks
       for block in self.blocks:
           x = block(x, t_emb, clip_emb)


       # Final processing
       combined_conditioning = torch.cat([t_emb, clip_emb], dim=1)
       shift, scale, gate, _, _, _ = self.final_adaLN(x, combined_conditioning)


       # Apply AdaLN to hidden features
       x = self.final_norm(x)
       x = x * (1 + scale) + shift
       x = gate * x


       # Project to output dimension
       x = self.final_proj(x)


       # Convert back to latent space
       x = self.unpatchify(x)


       return x




class CompositeLoss(nn.Module):
   """Fixed composite loss function"""


   def __init__(self, lambda_rec=1.0, lambda_lpips=1.0, lambda_clip=1.0, device='cuda'):
       super().__init__()
       self.lambda_rec = lambda_rec
       self.lambda_lpips = lambda_lpips
       self.lambda_clip = lambda_clip


       # Initialize LPIPS loss
       self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
       self.lpips_loss.eval()
       for param in self.lpips_loss.parameters():
           param.requires_grad = False


       # Initialize a separate CLIP model for loss computation
       self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
       self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


       # Keep CLIP frozen for loss computation
       for param in self.clip_model.parameters():
           param.requires_grad = False
       self.clip_model.eval()


   def reconstruction_loss(self, generated, target):
       """L1 reconstruction loss"""
       return F.l1_loss(generated, target)


   def perceptual_loss(self, generated, target):
       """LPIPS perceptual loss"""
       # Ensure images are in [-1, 1] range and require grad
       generated = torch.clamp(generated, -1, 1)
       target = torch.clamp(target, -1, 1)
       return self.lpips_loss(generated, target).mean()


   def clip_similarity_loss(self, generated, source):
       """CLIP semantic consistency loss - detached computation"""
       with torch.no_grad():
           # Convert generated images to CLIP format
           generated_clip = self.convert_to_clip_format(generated.detach())


           # Get CLIP features
           generated_features = self.clip_model.get_image_features(generated_clip)
           source_features = self.clip_model.get_image_features(source)


           # Normalize features
           generated_features = F.normalize(generated_features, dim=-1)
           source_features = F.normalize(source_features, dim=-1)


           # Cosine similarity
           similarity = torch.sum(generated_features * source_features, dim=-1)


           # Convert to loss (1 - cosine_similarity)
           loss_value = 1.0 - similarity.mean()


       # Create a tensor that requires grad for backprop
       loss_tensor = torch.tensor(loss_value.item(), device=generated.device, requires_grad=True)


       return loss_tensor


   def forward(self, generated, target, source):
       """
       Compute composite loss
       """
       # Reconstruction loss (L1) - this preserves gradients
       l_rec = self.reconstruction_loss(generated, target)


       # Perceptual loss (LPIPS) - this preserves gradients
       l_lpips = self.perceptual_loss(generated, target)


       # CLIP semantic loss - using detached computation
       l_clip = self.clip_similarity_loss(generated, source)


       # Composite loss - only L1 and LPIPS contribute to gradients
       total_loss = (self.lambda_rec * l_rec + self.lambda_lpips * l_lpips)


       # Add CLIP loss for monitoring
       total_loss = total_loss + self.lambda_clip * l_clip


       return {
           'total_loss': total_loss,
           'l_rec': l_rec.item(),
           'l_lpips': l_lpips.item(),
           'l_clip': l_clip.item()
       }


   def convert_to_clip_format(self, images):
       """Convert VAE format images to CLIP format"""
       # Convert from [-1, 1] to [0, 1]
       images = (images + 1.0) / 2.0
       images = torch.clamp(images, 0, 1)


       # Resize to 224x224 for CLIP
       images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)


       # Normalize for CLIP
       mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
       std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
       images = (images - mean) / std


       return images




class CLIPDiTTrainer:
   """Fixed training pipeline for CLIP-conditioned DiT"""


   def __init__(self, model, vae, noise_scheduler, optimizer, device,
                use_composite_loss=False, lambda_rec=1.0, lambda_lpips=0.5, lambda_clip=0.1):
       self.model = model
       self.vae = vae
       self.noise_scheduler = noise_scheduler
       self.optimizer = optimizer
       self.device = device
       self.use_composite_loss = use_composite_loss


       # Initialize composite loss if requested
       if use_composite_loss:
           self.composite_loss = CompositeLoss(
               lambda_rec=lambda_rec,
               lambda_lpips=lambda_lpips,
               lambda_clip=lambda_clip,
               device=device
           )


       # Image preprocessing
       self.vae_transform = transforms.Compose([
           transforms.Resize((256, 256)),
           transforms.ToTensor(),
           transforms.Normalize([0.5], [0.5])
       ])


       self.clip_transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])


   def encode_images(self, images):
       """Encode images using VAE"""
       with torch.no_grad():
           latents = self.vae.encode(images).latent_dist.sample()
           latents = latents * self.vae.config.scaling_factor
       return latents


   def decode_latents(self, latents):
       """Decode latents using VAE"""
       latents = latents / self.vae.config.scaling_factor
       with torch.no_grad():
           images = self.vae.decode(latents).sample
       return images


   def train_step(self, source_images, target_images):
       """Training step with proper gradient handling"""
       self.model.train()
       self.optimizer.zero_grad()


       try:
           # Encode target images to latents
           target_latents = self.encode_images(target_images)


           # Sample noise and timesteps
           noise = torch.randn_like(target_latents)
           timesteps = torch.randint(
               0, self.noise_scheduler.config.num_train_timesteps,
               (target_latents.shape[0],), device=self.device
           ).long()


           # Add noise to latents
           noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)


           # Predict noise - this is the main diffusion loss
           noise_pred = self.model(noisy_latents, timesteps, source_images)


           # Main diffusion loss (MSE between predicted and actual noise)
           diffusion_loss = F.mse_loss(noise_pred, noise, reduction='mean')


           total_loss = diffusion_loss
           loss_dict = {
               'total_loss': total_loss.item(),
               'diffusion_loss': diffusion_loss.item(),
           }


           # Optionally add composite loss
           if self.use_composite_loss and hasattr(self, 'composite_loss'):
               # Generate clean images for composite loss
               with torch.no_grad():
                   # Estimate x0 from noise prediction
                   alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps]
                   beta_prod_t = 1 - alpha_prod_t


                   alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
                   beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)


                   # Predict original latents
                   pred_original_latents = (noisy_latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                   pred_original_latents = torch.clamp(pred_original_latents, -4.0, 4.0)


               # Decode for composite loss
               generated_images = self.decode_latents(pred_original_latents)


               # Compute composite loss
               composite_loss_dict = self.composite_loss(
                   generated=generated_images,
                   target=target_images,
                   source=source_images
               )


               # Add a small amount of composite loss
               composite_loss_scaled = composite_loss_dict['total_loss'] * 0.1
               total_loss = diffusion_loss + composite_loss_scaled


               # Update loss dict
               loss_dict.update({
                   'total_loss': total_loss.item(),
                   'composite_loss': composite_loss_dict['total_loss'].item(),
                   'l_rec': composite_loss_dict['l_rec'],
                   'l_lpips': composite_loss_dict['l_lpips'],
                   'l_clip': composite_loss_dict['l_clip']
               })


           # Backward pass
           total_loss.backward()


           # Gradient clipping
           torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


           self.optimizer.step()


           return loss_dict


       except Exception as e:
           print(f"Error in train_step: {e}")
           # Return a default loss dict to prevent training from crashing
           return {'total_loss': float('inf'), 'diffusion_loss': float('inf')}


   def inference(self, source_images, num_inference_steps=20):
       """Generate images from source images"""
       self.model.eval()


       with torch.no_grad():
           # Start with random noise
           batch_size = source_images.shape[0]
           latents = torch.randn(
               batch_size, 4, 32, 32, device=self.device
           ) * self.noise_scheduler.init_noise_sigma


           # Set timesteps
           self.noise_scheduler.set_timesteps(num_inference_steps)


           # Denoise iteratively
           for i, t in enumerate(self.noise_scheduler.timesteps):
               t_batch = t.expand(batch_size).to(self.device)


               # Predict noise
               noise_pred = self.model(latents, t_batch, source_images)


               # Compute previous noisy sample
               latents = self.noise_scheduler.step(
                   noise_pred, t, latents, return_dict=False
               )[0]


           # Decode latents to images
           generated_images = self.decode_latents(latents)


       return generated_images




class Face2ComicsDataset(Dataset):
   """Dataset for face2comics with separate folders"""


   def __init__(self, face_dir, comic_dir, transform_vae=None, transform_clip=None):
       self.face_dir = face_dir
       self.comic_dir = comic_dir
       self.transform_vae = transform_vae
       self.transform_clip = transform_clip


       # Get matching filenames
       self.face_files = sorted([f for f in os.listdir(face_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
       self.comic_files = sorted([f for f in os.listdir(comic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


       # Ensure we have matching pairs
       assert len(self.face_files) == len(self.comic_files), "Mismatch in number of face and comic images"


       # Limit dataset size if specified
       if 450 < len(self.face_files):
           # Set seed for reproducible sampling
           random.seed(42)
           indices = random.sample(range(len(self.face_files)), 450)
           indices.sort()  # Keep sorted for consistency


           self.face_files = [self.face_files[i] for i in indices]
           self.comic_files = [self.comic_files[i] for i in indices]


       print(f"Dataset initialized with {len(self.face_files)} samples")


   def __len__(self):
       return len(self.face_files)


   def __getitem__(self, idx):
       # Load face (source) and comic (target)
       face_path = os.path.join(self.face_dir, self.face_files[idx])
       comic_path = os.path.join(self.comic_dir, self.comic_files[idx])


       face_img = Image.open(face_path).convert('RGB')
       comic_img = Image.open(comic_path).convert('RGB')


       # Apply transforms
       if self.transform_clip:
           face_img_clip = self.transform_clip(face_img)
       if self.transform_vae:
           comic_img_vae = self.transform_vae(comic_img)


       return face_img_clip, comic_img_vae




def create_model_and_trainer(device='cuda', use_composite_loss=False):
   """Create model and trainer with fixed gradient handling"""


   # Initialize model
   model = CLIPConditionedDiT(
       latent_size=32,
       patch_size=2,
       in_channels=4,
       hidden_size=768,
       depth=8,
       num_heads=12,
       mlp_ratio=4.0,
       clip_conditioning_size=768
   ).to(device)


   # Initialize VAE
   vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
   vae.requires_grad_(False)


   # Noise scheduler
   noise_scheduler = DDIMScheduler(
       num_train_timesteps=1000,
       beta_start=0.00085,
       beta_end=0.012,
       beta_schedule="scaled_linear",
       clip_sample=False,
       set_alpha_to_one=False,
       steps_offset=1,
   )


   # Optimizer
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=1e-4,
       weight_decay=1e-4,
       betas=(0.9, 0.999)
   )


   # Create fixed trainer
   trainer = CLIPDiTTrainer(
       model=model,
       vae=vae,
       noise_scheduler=noise_scheduler,
       optimizer=optimizer,
       device=device,
       use_composite_loss=use_composite_loss,
       lambda_rec=1.0,
       lambda_lpips=0.5,
       lambda_clip=0.1
   )


   return model, trainer




def create_dataloaders(dataset_type='face2comics', data_path='./data', batch_size=4, num_workers=2):
   """Create dataloaders for training"""
   # VAE transform for target images
   vae_transform = transforms.Compose([
       transforms.Resize((256, 256)),
       transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5])
   ])


   # CLIP transform for source images
   clip_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])


   if dataset_type == 'face2comics':
       train_dataset = Face2ComicsDataset(
           face_dir=os.path.join(data_path, 'faces'),
           comic_dir=os.path.join(data_path, 'comics'),
           transform_vae=vae_transform,
           transform_clip=clip_transform
       )
   else:
       raise ValueError("Only 'face2comics' dataset implemented")


   train_loader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=num_workers,
       pin_memory=True,
       drop_last=True
   )


   return train_loader




def save_checkpoint(model, optimizer, epoch, loss, path):
   """Save model checkpoint"""
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }, path)




def generate_samples(trainer, train_loader, epoch, save_dir, num_samples=4):
   """Generate sample images during training"""
   print(f"Generating samples for epoch {epoch + 1}...")


   # Get a batch of source images
   source_batch, _ = next(iter(train_loader))
   source_images = source_batch[:num_samples].to(trainer.device)


   # Generate images
   generated_images = trainer.inference(source_images, num_inference_steps=20)


   # Denormalize images for visualization
   def denormalize_clip(tensor):
       # Denormalize CLIP images
       mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
       std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
       return tensor * std + mean


   def denormalize_vae(tensor):
       # Denormalize VAE images from [-1, 1] to [0, 1]
       return (tensor + 1.0) / 2.0


   # Convert to numpy for visualization
   source_np = denormalize_clip(source_images).clamp(0, 1).cpu().numpy()
   generated_np = denormalize_vae(generated_images).clamp(0, 1).cpu().numpy()


   # Create comparison grid
   fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))


   for i in range(num_samples):
       # Source images
       axes[0, i].imshow(source_np[i].transpose(1, 2, 0))
       axes[0, i].set_title(f'Source {i + 1}')
       axes[0, i].axis('off')


       # Generated images
       axes[1, i].imshow(generated_np[i].transpose(1, 2, 0))
       axes[1, i].set_title(f'Generated {i + 1}')
       axes[1, i].axis('off')


   plt.tight_layout()
   os.makedirs(save_dir, exist_ok=True)
   plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch + 1}.png'))
   plt.close()




def create_model_and_trainer_composite_only(device='cuda'):
   """Create model and trainer with composite loss only"""


   # Initialize model
   model = CLIPConditionedDiT(
       latent_size=32,
       patch_size=2,
       in_channels=4,
       hidden_size=768,
       depth=8,
       num_heads=12,
       mlp_ratio=4.0,
       clip_conditioning_size=768
   ).to(device)


   # Initialize VAE
   vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
   vae.requires_grad_(False)


   # Noise scheduler
   noise_scheduler = DDIMScheduler(
       num_train_timesteps=1000,
       beta_start=0.00085,
       beta_end=0.012,
       beta_schedule="scaled_linear",
       clip_sample=False,
       set_alpha_to_one=False,
       steps_offset=1,
   )


   # Optimizer
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=1e-4,
       weight_decay=1e-4,
       betas=(0.9, 0.999)
   )


   # Create trainer with composite loss only
   trainer = CLIPDiTTrainer(
       model=model,
       vae=vae,
       noise_scheduler=noise_scheduler,
       optimizer=optimizer,
       device=device,
       use_composite_loss=True,  # Enable composite loss
       lambda_rec=1.0,
       lambda_lpips=0.5,
       lambda_clip=0.1
   )


   return model, trainer




class CompositeOnlyTrainer(CLIPDiTTrainer):
   """Trainer that uses only composite loss (no diffusion loss)"""


   def train_step(self, source_images, target_images):
       """Training step with composite loss only"""
       self.model.train()
       self.optimizer.zero_grad()


       try:
           # Encode target images to latents
           target_latents = self.encode_images(target_images)


           # Sample noise and timesteps
           noise = torch.randn_like(target_latents)
           timesteps = torch.randint(
               0, self.noise_scheduler.config.num_train_timesteps,
               (target_latents.shape[0],), device=self.device
           ).long()


           # Add noise to latents
           noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)


           # Predict noise
           noise_pred = self.model(noisy_latents, timesteps, source_images)


           # Estimate clean latents from noise prediction
           alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps]
           beta_prod_t = 1 - alpha_prod_t


           alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
           beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)


           # Predict original latents
           pred_original_latents = (noisy_latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
           pred_original_latents = torch.clamp(pred_original_latents, -4.0, 4.0)


           # Decode for composite loss
           generated_images = self.decode_latents(pred_original_latents)


           # Compute composite loss only
           composite_loss_dict = self.composite_loss(
               generated=generated_images,
               target=target_images,
               source=source_images
           )


           total_loss = composite_loss_dict['total_loss']


           # Backward pass
           total_loss.backward()


           # Gradient clipping
           torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


           self.optimizer.step()


           return {
               'total_loss': total_loss.item(),
               'l_rec': composite_loss_dict['l_rec'],
               'l_lpips': composite_loss_dict['l_lpips'],
               'l_clip': composite_loss_dict['l_clip']
           }


       except Exception as e:
           print(f"Error in train_step: {e}")
           return {'total_loss': float('inf'), 'l_rec': float('inf'), 'l_lpips': float('inf'), 'l_clip': float('inf')}




def train_loop(trainer, train_loader, num_epochs=10, save_dir='./checkpoints', sample_dir='./samples'):
   """Main training loop with composite loss only"""


   os.makedirs(save_dir, exist_ok=True)
   os.makedirs(sample_dir, exist_ok=True)


   best_loss = float('inf')


   for epoch in range(num_epochs):
       print(f"\nEpoch {epoch + 1}/{num_epochs}")


       # Training
       trainer.model.train()
       epoch_losses = []
       epoch_rec_losses = []
       epoch_lpips_losses = []
       epoch_clip_losses = []


       progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")


       for batch_idx, (source_images, target_images) in enumerate(progress_bar):
           source_images = source_images.to(trainer.device)
           target_images = target_images.to(trainer.device)


           # Training step
           loss_dict = trainer.train_step(source_images, target_images)


           epoch_losses.append(loss_dict['total_loss'])
           epoch_rec_losses.append(loss_dict['l_rec'])
           epoch_lpips_losses.append(loss_dict['l_lpips'])
           epoch_clip_losses.append(loss_dict['l_clip'])


           # Update progress bar
           progress_bar.set_postfix({
               'Loss': f"{loss_dict['total_loss']:.4f}",
               'L_rec': f"{loss_dict['l_rec']:.4f}",
               'L_lpips': f"{loss_dict['l_lpips']:.4f}",
               'L_clip': f"{loss_dict['l_clip']:.4f}"
           })


       # Calculate average losses
       avg_loss = np.mean(epoch_losses)
       avg_rec_loss = np.mean(epoch_rec_losses)
       avg_lpips_loss = np.mean(epoch_lpips_losses)
       avg_clip_loss = np.mean(epoch_clip_losses)


       print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, "
             f"L_rec: {avg_rec_loss:.4f}, L_lpips: {avg_lpips_loss:.4f}, L_clip: {avg_clip_loss:.4f}")


       # Generate samples every 5 epochs
       if (epoch + 1) % 5 == 0:
           generate_samples(trainer, train_loader, epoch, sample_dir)


       # Save checkpoint if best loss
       if avg_loss < best_loss:
           best_loss = avg_loss
           save_checkpoint(
               trainer.model,
               trainer.optimizer,
               epoch,
               avg_loss,
               os.path.join(save_dir, 'best_model.pth')
           )
           print(f"Saved best model with loss: {best_loss:.4f}")


       # Save regular checkpoint every 10 epochs
       if (epoch + 1) % 10 == 0:
           save_checkpoint(
               trainer.model,
               trainer.optimizer,
               epoch,
               avg_loss,
               os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
           )


def plot_training_losses(loss_history, save_path='./plots/training_losses.png'):
   """
   Plot training losses over epochs


   Args:
       loss_history: Dictionary containing loss arrays for each epoch
       save_path: Path to save the plot
   """
   os.makedirs(os.path.dirname(save_path), exist_ok=True)


   epochs = range(1, len(loss_history['total_loss']) + 1)


   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   fig.suptitle('Training Loss Over Epochs', fontsize=16)


   # Total Loss
   axes[0, 0].plot(epochs, loss_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
   axes[0, 0].set_title('Total Loss')
   axes[0, 0].set_xlabel('Epoch')
   axes[0, 0].set_ylabel('Loss')
   axes[0, 0].grid(True, alpha=0.3)
   axes[0, 0].legend()


   # Reconstruction Loss
   axes[0, 1].plot(epochs, loss_history['l_rec'], 'r-', linewidth=2, label='L1 Reconstruction')
   axes[0, 1].set_title('Reconstruction Loss (L1)')
   axes[0, 1].set_xlabel('Epoch')
   axes[0, 1].set_ylabel('Loss')
   axes[0, 1].grid(True, alpha=0.3)
   axes[0, 1].legend()


   # LPIPS Loss
   axes[1, 0].plot(epochs, loss_history['l_lpips'], 'g-', linewidth=2, label='LPIPS Perceptual')
   axes[1, 0].set_title('Perceptual Loss (LPIPS)')
   axes[1, 0].set_xlabel('Epoch')
   axes[1, 0].set_ylabel('Loss')
   axes[1, 0].grid(True, alpha=0.3)
   axes[1, 0].legend()


   # CLIP Loss
   axes[1, 1].plot(epochs, loss_history['l_clip'], 'm-', linewidth=2, label='CLIP Semantic')
   axes[1, 1].set_title('CLIP Semantic Loss')
   axes[1, 1].set_xlabel('Epoch')
   axes[1, 1].set_ylabel('Loss')
   axes[1, 1].grid(True, alpha=0.3)
   axes[1, 1].legend()


   plt.tight_layout()
   plt.savefig(save_path, dpi=300, bbox_inches='tight')
   plt.close()
   print(f"Training loss plot saved to: {save_path}")




def plot_combined_losses(loss_history, save_path='./plots/combined_losses.png'):
   """
   Plot all losses on the same graph with different y-axes
   """
   os.makedirs(os.path.dirname(save_path), exist_ok=True)


   epochs = range(1, len(loss_history['total_loss']) + 1)


   fig, ax1 = plt.subplots(figsize=(10, 6))


   # Plot total loss on primary y-axis
   color = 'tab:blue'
   ax1.set_xlabel('Epoch')
   ax1.set_ylabel('Total Loss', color=color)
   ax1.plot(epochs, loss_history['total_loss'], color=color, linewidth=2, label='Total Loss')
   ax1.tick_params(axis='y', labelcolor=color)
   ax1.grid(True, alpha=0.3)


   # Create secondary y-axis for component losses
   ax2 = ax1.twinx()
   ax2.set_ylabel('Component Losses')
   ax2.plot(epochs, loss_history['l_rec'], 'r--', linewidth=1.5, label='L1 Reconstruction')
   ax2.plot(epochs, loss_history['l_lpips'], 'g--', linewidth=1.5, label='LPIPS Perceptual')
   ax2.plot(epochs, loss_history['l_clip'], 'm--', linewidth=1.5, label='CLIP Semantic')


   # Add legends
   lines1, labels1 = ax1.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')


   plt.title('Training Losses Over Epochs')
   plt.tight_layout()
   plt.savefig(save_path, dpi=300, bbox_inches='tight')
   plt.close()
   print(f"Combined loss plot saved to: {save_path}")




def save_loss_history(loss_history, save_path='./plots/loss_history.json'):
   """Save loss history to JSON file for later analysis"""
   os.makedirs(os.path.dirname(save_path), exist_ok=True)


   with open(save_path, 'w') as f:
       json.dump(loss_history, f, indent=2)
   print(f"Loss history saved to: {save_path}")




def load_loss_history(load_path='./plots/loss_history.json'):
   """Load loss history from JSON file"""
   with open(load_path, 'r') as f:
       return json.load(f)




# Modified training loop with loss tracking
def train_loop_with_plotting(trainer, train_loader, num_epochs=10, save_dir='./checkpoints',
                            sample_dir='./samples', plot_dir='./plots'):
   """Enhanced training loop with loss plotting"""


   os.makedirs(save_dir, exist_ok=True)
   os.makedirs(sample_dir, exist_ok=True)
   os.makedirs(plot_dir, exist_ok=True)


   # Initialize loss history
   loss_history = {
       'total_loss': [],
       'l_rec': [],
       'l_lpips': [],
       'l_clip': []
   }


   best_loss = float('inf')


   for epoch in range(num_epochs):
       print(f"\nEpoch {epoch + 1}/{num_epochs}")


       # Training
       trainer.model.train()
       epoch_losses = []
       epoch_rec_losses = []
       epoch_lpips_losses = []
       epoch_clip_losses = []


       progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")


       for batch_idx, (source_images, target_images) in enumerate(progress_bar):
           source_images = source_images.to(trainer.device)
           target_images = target_images.to(trainer.device)


           # Training step
           loss_dict = trainer.train_step(source_images, target_images)


           epoch_losses.append(loss_dict['total_loss'])
           epoch_rec_losses.append(loss_dict['l_rec'])
           epoch_lpips_losses.append(loss_dict['l_lpips'])
           epoch_clip_losses.append(loss_dict['l_clip'])


           # Update progress bar
           progress_bar.set_postfix({
               'Loss': f"{loss_dict['total_loss']:.4f}",
               'L_rec': f"{loss_dict['l_rec']:.4f}",
               'L_lpips': f"{loss_dict['l_lpips']:.4f}",
               'L_clip': f"{loss_dict['l_clip']:.4f}"
           })


       # Calculate average losses for the epoch
       avg_loss = np.mean(epoch_losses)
       avg_rec_loss = np.mean(epoch_rec_losses)
       avg_lpips_loss = np.mean(epoch_lpips_losses)
       avg_clip_loss = np.mean(epoch_clip_losses)


       # Store in loss history
       loss_history['total_loss'].append(avg_loss)
       loss_history['l_rec'].append(avg_rec_loss)
       loss_history['l_lpips'].append(avg_lpips_loss)
       loss_history['l_clip'].append(avg_clip_loss)


       print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, "
             f"L_rec: {avg_rec_loss:.4f}, L_lpips: {avg_lpips_loss:.4f}, L_clip: {avg_clip_loss:.4f}")


       # Plot losses every 5 epochs or at the last epoch
       if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
           plot_training_losses(loss_history, os.path.join(plot_dir, f'training_losses_epoch_{epoch + 1}.png'))
           plot_combined_losses(loss_history, os.path.join(plot_dir, f'combined_losses_epoch_{epoch + 1}.png'))
           save_loss_history(loss_history, os.path.join(plot_dir, 'loss_history.json'))


       # Generate samples every 5 epochs
       if (epoch + 1) % 5 == 0:
           generate_samples(trainer, train_loader, epoch, sample_dir)


       # Save checkpoint if best loss
       if avg_loss < best_loss:
           best_loss = avg_loss
           save_checkpoint(
               trainer.model,
               trainer.optimizer,
               epoch,
               avg_loss,
               os.path.join(save_dir, 'best_model.pth')
           )
           print(f"Saved best model with loss: {best_loss:.4f}")


       # Save regular checkpoint every 10 epochs
       if (epoch + 1) % 10 == 0:
           save_checkpoint(
               trainer.model,
               trainer.optimizer,
               epoch,
               avg_loss,
               os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
           )


   # Final plots
   plot_training_losses(loss_history, os.path.join(plot_dir, 'final_training_losses.png'))
   plot_combined_losses(loss_history, os.path.join(plot_dir, 'final_combined_losses.png'))
   save_loss_history(loss_history, os.path.join(plot_dir, 'final_loss_history.json'))


   return loss_history




# Function to plot from saved history (useful for analysis later)
def plot_from_saved_history(history_path='./plots/loss_history.json', output_dir='./plots'):
   """Load and plot from saved loss history"""
   loss_history = load_loss_history(history_path)
   plot_training_losses(loss_history, os.path.join(output_dir, 'replotted_training_losses.png'))
   plot_combined_losses(loss_history, os.path.join(output_dir, 'replotted_combined_losses.png'))


def main():
   """Main training function with loss plotting"""


   # Configuration
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   print(f"Using device: {device}")


   # Data paths - adjust these to your dataset
   data_path = './'  # Should contain 'faces' and 'comics' folders


   # Training parameters
   batch_size = 4
   num_epochs = 10  # change if needed
   num_workers = 2


   # Create directories
   save_dir = './checkpoints'
   sample_dir = './samples'
   plot_dir = './plots'  # New directory for plots
   os.makedirs(save_dir, exist_ok=True)
   os.makedirs(sample_dir, exist_ok=True)
   os.makedirs(plot_dir, exist_ok=True)


   try:
       # Create dataloaders
       print("Creating dataloaders...")
       train_loader = create_dataloaders(
           dataset_type='face2comics',
           data_path=data_path,
           batch_size=batch_size,
           num_workers=num_workers
       )
       print(f"Dataset size: {len(train_loader.dataset)} samples")


       # Create model and trainer (composite loss only)
       print("Creating model and trainer...")
       model, base_trainer = create_model_and_trainer_composite_only(device=device)


       # Use composite-only trainer
       trainer = CompositeOnlyTrainer(
           model=base_trainer.model,
           vae=base_trainer.vae,
           noise_scheduler=base_trainer.noise_scheduler,
           optimizer=base_trainer.optimizer,
           device=device,
           use_composite_loss=True,
           lambda_rec=1.0,
           lambda_lpips=0.5,
           lambda_clip=0.1
       )


       print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


       # Start training with plotting
       print("Starting training with loss plotting...")
       loss_history = train_loop_with_plotting(
           trainer=trainer,
           train_loader=train_loader,
           num_epochs=num_epochs,
           save_dir=save_dir,
           sample_dir=sample_dir,
           plot_dir=plot_dir
       )


       print("Training completed!")
       print(f"Final losses - Total: {loss_history['total_loss'][-1]:.4f}, "
             f"L_rec: {loss_history['l_rec'][-1]:.4f}, "
             f"L_lpips: {loss_history['l_lpips'][-1]:.4f}, "
             f"L_clip: {loss_history['l_clip'][-1]:.4f}")


   except Exception as e:
       print(f"Error in main: {e}")
       import traceback
       traceback.print_exc()


if __name__ == "__main__":
   main()











