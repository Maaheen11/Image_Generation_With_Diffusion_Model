## Image Generation With Denoising Diffusion Probabilistic Model (DDPM)

This project implements a simplified version of Denoising Diffusion Probabilistic Models (DDPM) from scratch in PyTorch, based on the original paper Denoising Diffusion Probabilistic Models by Ho et al. The model is trained on the AFHQ dataset to generate realistic images of animals (cats, dogs, wildlife) by progressively denoising Gaussian noise.

## Features

- End-to-end PyTorch implementation of the DDPM  
- Forward and reverse diffusion processes  
- Support for linear and quadratic noise schedules  
- U-Net-based noise prediction with sinusoidal timestep embeddings  
- MSE-based loss function for noise prediction  
- Image sampling from Gaussian noise  
- Evaluation using Frechet Inception Distance (FID)

## Sample Outputs

Generated images from trained model (T = 98000 steps):

![913](https://github.com/user-attachments/assets/4ab89102-cada-4826-ba2b-41c480fd1aee)

![989](https://github.com/user-attachments/assets/d8bd1ecc-24a5-42c0-9bb8-a9a86affadee)

![923](https://github.com/user-attachments/assets/1d542079-5af3-4fce-ba29-280a8c00d4e4)

## Repository Structure
├── code files/ # All code files for DDPM Implementation
├── Results/ # Generated samples during evaluation
├── data # partial AFHQ dataset
├── Original DDPM Research Paper # DDPM Paper by Ho et al
├── training logs # Configuration JSON file, 4 Model Results at Step t 
├── FID.png # Computed FID during evaluation
├── loss.png # Training Loss Plot
└── README.md

## Training the Model 
python train.py 

## Evaluating the Model 

Generating Images 
python evaluate.py sample --ckpt_path last.ckpt --save_path results/


Computing FID
python evaluate.py fid --ckpt_path last.ckpt
