# Balance Scale

A PyTorch research notebook for classifying skin lesions across 11 diagnostic categories using the MILK10k dataset. It explores combining clinical and dermoscopic images with their associated metadata, while addressing class imbalance during training.

## Approach

Each lesion provides four inputs: a dermoscopic image, a clinical close-up image, and a metadata vector for each view. Metadata includes age, sex, skin tone, anatomical site, and selected MONET image attributes.

- A shared pretrained **EfficientNetV2-S** backbone extracts image features.
- A metadata network encodes imputed, scaled, and one-hot-encoded attributes.
- **Gated Multimodal Units** combine each image with its metadata, then combine both views.
- A main classifier and two auxiliary heads are trained with asymmetric focal loss. Weighted random sampling increases exposure to underrepresented classes.

Class columns: `AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`.

## Run the notebook

1. Import `experiments.ipynb` into Kaggle.
2. Attach the [MILK10k dataset](https://www.kaggle.com/datasets/hamzabinbutt/isic-challenge-datasets).
3. Enable a CUDA GPU accelerator and internet access for downloading pretrained weights.
4. Ensure PyTorch, torchvision, timm, NumPy, pandas, Pillow, and scikit-learn are available.
5. Check the paths below, then run all cells in order.

```python
DATASET_DIR = '/kaggle/input/datasets/hamzabinbutt/isic-challenge-datasets'
SAVE_DIR = '/kaggle/working'
```

The dataset root must contain the training ground-truth CSV, training/test metadata CSVs, and image folders. The notebook expects nested image directories named `MILK10k_Training_Input/MILK10k_Training_Input` and `MILK10k_Test_Input/MILK10k_Test_Input`, with images grouped by lesion ID.

## Training and evaluation

| Setting | Default |
| --- | --- |
| Split | 85% training, 15% validation; stratified by class at lesion level |
| Image size / batch size | 380 × 380 / 16 |
| Random seed | 42 |
| Phase 1 | 15 epochs with frozen backbone parameters |
| Phase 2 | 40 epochs selectively fine-tuning later backbone layers |
| Validation metric | Macro-F1 with a fixed 0.5 threshold on sigmoid probabilities |

Training uses AdamW, learning-rate scheduling, image augmentation, CUDA mixed precision, and gradient clipping. The checkpoint with the highest validation macro-F1 is used for test predictions.

## Outputs

Files are written to `SAVE_DIR`:

| File | Contents |
| --- | --- |
| `milk10k_pt_gmu_best.pt` | Best model's PyTorch state dictionary |
| `milk10k_pt_gmu_submission.csv` | An `image` column containing lesion IDs, followed by 11 class-probability columns |

## Repository scope

- `experiments.ipynb` — preprocessing, model, training, validation, and submission generation.
- `README.md` — original project overview and challenge links.

This is an evolving experiment. The committed notebook contains no saved cell outputs or numerical validation results; datasets, trained checkpoints, and submissions are not included. Dependencies are not pinned, and the preprocessing pipeline is not saved with the model weights. Reusing a checkpoint requires recreating compatible preprocessing and model configuration.
