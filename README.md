# SAM2 Model Optimization and Benchmarking Project

## Overview
This project focuses on fine-tuning the Segment Anything Model 2 (SAM2) using publicly available datasets, implementing optimization techniques like quantization and pruning to reduce model size, and benchmarking the performance against the SegFormer model. The entire pipeline is executed on an RTX 4070 Mobile GPU.

## Project Structure
The project consists of several key components:

1. **Fine-tuning Pipeline**
   - `SAM2Finetuning.ipynb` and `SAM2Finetuning2.ipynb`: Implementation of fine-tuning process using Cityscapes dataset
   - `Finetuning.ipynb`: Additional fine-tuning experiments

2. **Optimization Pipeline**
   - `sam2optimisation.ipynb`: Implementation of quantization and pruning techniques
   - `bm_optimized_sam2.ipynb`: Benchmarking of the optimized model

3. **Benchmarking Pipeline**
   - `benchmarking_segformer.ipynb`: Performance evaluation of SegFormer model
   - `BenchmarkingSAM2.ipynb` and `BenchmarkingSAM2_video.ipynb`: Performance evaluation of SAM2
   - `BenchmarkingFinetunedSAM2.ipynb`: Performance evaluation of fine-tuned SAM2

4. **Visualization and Analysis**
   - `visualisation.ipynb`: Tools for visualizing model outputs and performance metrics

## Implementation Details

### Fine-tuning Strategy
- Utilized Cityscapes dataset for fine-tuning
- Implemented a 10-epoch training cycle
- Achieved progressive improvement in Mean IoU from 0.0033 to 0.2953 during training
- Used standard segmentation loss functions and optimization techniques

### Optimization Techniques
1. **Quantization**
   - Implemented post-training quantization
   - Used calibration dataset of 500 images
   - Reduced model precision while maintaining performance

2. **Pruning**
   - Applied structured pruning to reduce model size
   - Implemented layer-wise pruning strategies
   - Optimized for inference speed and memory usage

### Benchmarking Results
The project compared three models:
1. Original SAM2
2. Optimized SAM2 (after quantization and pruning)
3. SegFormer

Key metrics measured:
- Pixel Accuracy
- Mean IoU
- FLOPs per forward pass
- Inference Time
- Frames Per Second (FPS)

## Hardware Specifications
- GPU: NVIDIA RTX 4070 Mobile
- Operating System: Windows 11
- Python Environment: Anaconda with PyTorch

## Dependencies
- PyTorch
- SAM2
- Transformers
- OpenCV
- NumPy
- Matplotlib

## Usage
1. Fine-tuning:
   ```python
   # Run SAM2Finetuning2.ipynb for fine-tuning process
   ```

2. Optimization:
   ```python
   # Run sam2optimisation.ipynb for quantization and pruning
   ```

3. Benchmarking:
   ```python
   # Run benchmarking_segformer.ipynb and BenchmarkingSAM2.ipynb for performance comparison
   ```

## Results and Analysis
The project successfully demonstrated:
- Effective fine-tuning of SAM2 on Cityscapes dataset
- Significant model size reduction through optimization techniques
- Comparative performance analysis with SegFormer
- Real-time performance metrics on mobile GPU

## Future Work
- Implementation of additional optimization techniques
- Extension to other datasets
- Further performance improvements
- Deployment optimization for edge devices

