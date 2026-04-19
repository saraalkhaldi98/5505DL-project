# Human activity classification (CS552J)

This project builds and compares several approaches to **classify people’s activities** from still images: simple neural baselines, convolutional models with augmentation and regularization, transfer learning with a pretrained image model, and CLIP-based embeddings and text–image similarity.

## Data

The catalogue for the images is **`cs552j_A1_dataset_image_id_url.csv`**. Each row describes one photograph from the MS COCO ecosystem: identifiers, dimensions, capture metadata, a download URL, and a **string label** for the activity (e.g. sitting, standing, walking or running). You download the actual JPEGs from the URLs in that table and keep them alongside your code so the training code can resolve paths from the `file_name` column.

The split used in the workflow is **stratified train/validation** so that class proportions stay similar in both subsets.

## What the workflow covers

- **Baselines:** a fully connected network on flattened pixels versus a small convolutional network trained from scratch on the same split and metrics.
- **Generalization:** training-time augmentation, dropout, weight decay, and early stopping on validation accuracy.
- **Transfer learning:** a compact pretrained vision model fine-tuned for a **binary** task (two chosen labels).
- **CLIP:** frozen image encoder with a small trainable head on L2-normalized embeddings; optional zero-shot and hybrid scoring using text prompts aligned to each class.

Metrics reported include **accuracy**, **macro-averaged F1**, and **confusion matrices** on the validation portion appropriate to each sub-experiment.

## Environment

Use **Python 3** with a recent **PyTorch** stack, **torchvision**, **pandas**, **NumPy**, **scikit-learn**, **Matplotlib**, **Pillow**, **requests**, **tqdm**, and **Hugging Face Transformers**

## Reproducibility

Keep a fixed **random seed** for NumPy, Python, and PyTorch, and record **hardware** (CPU/GPU), **library versions**, **split ratio**, and **hyperparameters** so results can be replayed.
