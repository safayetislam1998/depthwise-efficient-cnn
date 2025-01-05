# Efficient CNN with Depthwise Separable Convolutions

## ⚠️ Version Compatibility Notice
This implementation was developed and tested specifically in the Kaggle TPU environment with:
- TensorFlow 2.10.0
- Keras 2.10.0
- TensorFlow Addons 

**Important:** 
1. This code was designed to run on TPU accelerators and was optimized for the Kaggle TPU runtime environment.
2. Due to significant API changes in newer versions of TensorFlow/Keras (especially Keras 3.0), the code requires specific version dependencies.
3. TensorFlow Addons (used for AdaptiveAveragePooling2D and AdamW) is now in maintenance mode and is no longer actively maintained. Future compatibility is not guaranteed.

## Project Structure
- ├── main.py          # Main training script and TPU/GPU setup
- ├── model.py         # Model architecture 
- ├── data.py         # Data and augmentation pipeline
- ├── trainer.py      # Training utilities and evaluation metrics
- └── augment.py      # Implementation of augmentation techniques (FROM TF Model Garden)

## Model Variants
Three model variants were developed with different speed-accuracy tradeoffs:

1. **Small Model**
   - Parameters: 130,442
   - FLOPS: 0.0325G
   - Accuracy: 93%

2. **Medium Model**
   - Parameters: 490,250
   - FLOPS: 0.1238G
   - Accuracy: 95%

3. **Large Model**
   - Parameters: 998,666
   - FLOPS: 1.0092G
   - Accuracy: 97%

## Results

### Classification Performance
- Best accuracy achieved: 97% on CIFAR-10
- Training conducted over 300 epochs
- Used combination of advanced augmentation techniques:
  - AutoAugment
  - RandAugment
  - Mixup
  - CutMix
