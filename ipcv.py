# Required Libraries: torch, torchvision, sklearn, matplotlib, numpy, tqdm, captum, Pillow
# Install them using: pip install torch torchvision scikit-learn matplotlib numpy tqdm captum Pillow

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
from tqdm import tqdm
import copy
from captum.attr import LRP # For Explainable AI (XAI) [cite: 12, 156]

# --- 1. Configuration & Setup ---

# Basic Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Parameters (Based on Paper [cite: 186, 188, 189]) ---
DATA_DIR = r'D:\rs\Monkeypox Skin Image Dataset' 
MODEL_SAVE_PATH = 'skin_disease_vgg16_lrp.pth'
NUM_CLASSES = 4 # Chickenpox, Measles, Monkeypox, Normal [cite: 8, 114]
BATCH_SIZE = 32 # [cite: 186]
NUM_EPOCHS = 10 # [cite: 186, 188]
LEARNING_RATE = 0.00001 # [cite: 186]
INPUT_SIZE = 224 # VGG16 expects 224x224 images [cite: 121, 146]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")
logger.info("Configuration loaded.")

# --- 2. Data Loading and Preprocessing (Modified for Auto-Split) ---
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import os
import logging
import numpy as np # Ensure numpy is imported if not already

# Setup logger if running this block standalone
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration (ensure these are defined or passed from Section 1)
 # IMPORTANT: Replace with the actual path to your dataset directory
INPUT_SIZE = 224
BATCH_SIZE = 32
# Define VAL_SPLIT based on paper's 80/20 train/test split [cite: 123]
VAL_SPLIT = 0.20

logger.info("Data loading configuration: Splitting dataset with validation size = {:.0%}".format(VAL_SPLIT))

# Define transformations
# Train transform includes augmentation
# Val transform does not include augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(), # Basic Augmentation
        transforms.RandomRotation(10),    # Basic Augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the *entire* dataset once with train transforms and once with val transforms
# This allows applying the correct augmentation after splitting the indices
logger.info("Loading full dataset (with train transforms)...")
try:
    full_dataset_train_tf = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    logger.info("Loading full dataset (with validation transforms)...")
    full_dataset_val_tf = datasets.ImageFolder(DATA_DIR, transform=data_transforms['val'])

    # Check if datasets loaded successfully and have the same size
    if len(full_dataset_train_tf) != len(full_dataset_val_tf):
         raise ValueError("Inconsistent dataset sizes after loading with different transforms.")
    if len(full_dataset_train_tf) == 0:
         raise ValueError("Dataset is empty. Check DATA_DIR path and contents.")

    class_names = full_dataset_train_tf.classes
    logger.info(f"Dataset classes found: {class_names}")
    NUM_CLASSES = len(class_names) # Update NUM_CLASSES based on dataset

except FileNotFoundError:
    logger.error(f"Error: Dataset directory not found at {DATA_DIR}. Please provide the correct path.")
    exit()
except ValueError as ve:
     logger.error(f"Error processing dataset: {ve}")
     exit()
except Exception as e:
    logger.error(f"An unexpected error occurred during dataset loading: {e}")
    exit()

# Calculate split sizes
total_size = len(full_dataset_train_tf)
val_size = int(np.floor(VAL_SPLIT * total_size))
train_size = total_size - val_size
logger.info(f"Total dataset size: {total_size}. Splitting into Train: {train_size}, Validation: {val_size}")

# Perform the random split to get indices
# Use a fixed generator for reproducibility if desired
train_indices, val_indices = random_split(range(total_size), [train_size, val_size])#, generator=torch.Generator().manual_seed(42))

# Create Subset objects using the *same* indices but *different* underlying datasets (with respective transforms)
train_dataset = Subset(full_dataset_train_tf, train_indices)
val_dataset = Subset(full_dataset_val_tf, val_indices)

# Create DataLoaders for the subsets
image_datasets = {'train': train_dataset, 'val': val_dataset}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True if x == 'train' else False, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

logger.info(f"DataLoaders created. Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")

# --- End of Modified Section 2 ---

# --- 3. Model Definition (VGG16 Transfer Learning) ---

logger.info("Loading pre-trained VGG16 model...")
# Load pre-trained VGG16 model [cite: 7, 144]
model_ft = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Freeze convolutional base layers (optional but common for transfer learning)
# for param in model_ft.features.parameters():
#    param.requires_grad = False

# Modify the final classifier layer for our number of classes [cite: 151]
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

model_ft = model_ft.to(DEVICE)
logger.info("Model loaded and modified for the task.")
# logger.info(model_ft) # Uncomment to print model structure

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
# Adam optimizer as specified in the paper [cite: 186]
optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

# --- 4. Training Function ---

def train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS):
    logger.info(f"Starting training for {num_epochs} epochs...")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f'--- Epoch {epoch+1}/{num_epochs} ---')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data with progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}", unit="batch")
            for inputs, labels in progress_bar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), accuracy=torch.sum(preds == labels.data).item() / inputs.size(0))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if best accuracy achieved in validation
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                logger.info(f"New best validation accuracy: {best_acc:.4f}. Model saved to {MODEL_SAVE_PATH}")

    logger.info(f'Training complete. Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# --- 5. Evaluation Function ---

def evaluate_model(model, dataloader):
    logger.info("Starting evaluation...")
    model.eval() # Set model to evaluate mode
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics (Based on paper Eqs. 1-8)
    accuracy = accuracy_score(all_labels, all_preds)
    # average='weighted' handles potential class imbalance better for overall scores
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    # Calculate multiclass confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Calculate per-class metrics and then overall metrics as in the paper
    # TP, TN, FP, FN calculation for multi-class from CM
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Overall metrics (micro-averaged, equivalent to overall accuracy components)
    overall_tp = TP.sum()
    overall_fp = FP.sum()
    overall_fn = FN.sum()
    overall_tn = TN.sum()
    total = overall_tp + overall_fp + overall_fn + overall_tn

    overall_accuracy = (overall_tp + overall_tn) / total if total > 0 else 0 # [cite: 168]
    misclassification_rate = (overall_fp + overall_fn) / total if total > 0 else 0 # [cite: 170]
    overall_sensitivity = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0 # Recall / TPR [cite: 178]
    overall_specificity = overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0 # [cite: 176]
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0 # [cite: 174]
    overall_fpr = overall_fp / (overall_fp + overall_tn) if (overall_fp + overall_tn) > 0 else 0 # [cite: 182]
    overall_fnr = overall_fn / (overall_fn + overall_tp) if (overall_fn + overall_tp) > 0 else 0 # [cite: 180]
    overall_f1 = 2 * (overall_precision * overall_sensitivity) / (overall_precision + overall_sensitivity) if (overall_precision + overall_sensitivity) > 0 else 0 # [cite: 184]


    logger.info(f"\n--- Evaluation Metrics ---")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f} (Matches paper's definition)")
    logger.info(f"Misclassification Rate: {misclassification_rate:.4f}")
    logger.info(f"Weighted Precision: {precision:.4f} (Overall Precision: {overall_precision:.4f})")
    logger.info(f"Weighted Recall (Sensitivity): {recall:.4f} (Overall Sensitivity/TPR: {overall_sensitivity:.4f})")
    logger.info(f"Overall Specificity: {overall_specificity:.4f}")
    logger.info(f"Overall False Positive Rate (FPR): {overall_fpr:.4f}")
    logger.info(f"Overall False Negative Rate (FNR): {overall_fnr:.4f}")
    logger.info(f"Weighted F1-Score: {f1:.4f} (Overall F1-Score: {overall_f1:.4f})")


    # Plot Confusion Matrix [cite: 185]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix saved to confusion_matrix.png")
    # plt.show() # Uncomment to display plot immediately

# --- 6. Explainable AI (LRP Visualization) ---

def visualize_lrp(model, dataloader, num_images=4):
    logger.info(f"Starting LRP visualization for {num_images} images...")
    model.eval()
    lrp = LRP(model) # Initialize LRP with the trained model [cite: 157, 197]

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    if images.shape[0] < num_images:
        num_images = images.shape[0]
        logger.warning(f"Requested {num_images} for LRP but batch only has {images.shape[0]}. Using {images.shape[0]}.")

    # Select images and calculate LRP attributions
    input_imgs = images[:num_images].requires_grad_() # Ensure requires_grad=True for LRP
    target_labels = labels[:num_images]

    try:
        # Calculate attribution for the target class
        attribution = lrp.attribute(input_imgs, target=target_labels)
        attribution = attribution.detach().cpu().numpy()
    except Exception as e:
        logger.error(f"Error during LRP attribution calculation: {e}")
        logger.error("LRP might require specific model adaptations or different parameters depending on layers used.")
        return

    # Prepare images for plotting (denormalize)
    input_imgs_plot = input_imgs.detach().cpu().numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_imgs_plot = std * input_imgs_plot + mean
    input_imgs_plot = np.clip(input_imgs_plot, 0, 1)

    # Sum attributions across color channels for visualization
    vis_attr = np.sum(attribution, axis=1)

    # Plot original images and LRP heatmaps [cite: 199, 200]
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(6, 3 * num_images))
    fig.suptitle('LRP Explanations', fontsize=16)

    for i in range(num_images):
        true_label = class_names[target_labels[i].item()]
        # Predict label for title
        with torch.no_grad():
             outputs = model(images[i:i+1]) # Predict for single image
             _, predicted_idx = torch.max(outputs, 1)
             predicted_label = class_names[predicted_idx.item()]

        # Original Image
        ax = axes[i, 0] if num_images > 1 else axes[0]
        ax.imshow(input_imgs_plot[i])
        ax.set_title(f'Original (True: {true_label}\nPred: {predicted_label})')
        ax.axis('off')

        # LRP Heatmap
        ax = axes[i, 1] if num_images > 1 else axes[1]
        im = ax.imshow(vis_attr[i], cmap='Reds') # Use Reds colormap for relevance
        ax.set_title('LRP Relevance')
        ax.axis('off')
        # Add colorbar - adjust placement as needed
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig('lrp_explanations.png')
    logger.info("LRP visualization saved to lrp_explanations.png")
    # plt.show() # Uncomment to display plot immediately


# --- 7. Main Execution ---

if __name__ == '__main__':
    logger.info("--- Starting Skin Disease Prediction Pipeline ---")

    # --- Train the model ---
    # model_trained = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)
    # logger.info("Training finished.")

    # --- OR Load a pre-trained model if available ---
    try:
        logger.info(f"Loading model weights from {MODEL_SAVE_PATH}")
        model_ft.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model_trained = model_ft
        logger.info("Model weights loaded successfully.")
    except FileNotFoundError:
        logger.warning(f"Model file {MODEL_SAVE_PATH} not found. Training the model first.")
        model_trained = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)
        logger.info("Training finished.")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}. Training the model.")
        model_trained = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)
        logger.info("Training finished.")


    # --- Evaluate the model ---
    evaluate_model(model_trained, dataloaders['val'])
    logger.info("Evaluation finished.")

    # --- Generate LRP Explanations ---
    visualize_lrp(model_trained, dataloaders['val'], num_images=4) # Visualize for 4 validation images
    logger.info("LRP visualization finished.")

    logger.info("--- Pipeline Finished ---")