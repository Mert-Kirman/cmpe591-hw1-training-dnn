# Robot Control: Vision-Based State Prediction and Reconstruction

This repository contains the implementation of three deep learning models designed to process raw visual states and physical actions in a simulated robotic environment. The goal is to predict the future state of objects on a table after a robotic arm executes a specific pushing action.

## Getting Started

### Usage Instructions

**Step 1: Generate Training Data**

Before training any models, you must first collect the dataset by running:
```bash
python data_collector.py
```
This will create a `data/` directory containing 1,000 simulated samples (4 parallel processes × 250 samples each). The collection process may take several minutes depending on your system.

**Step 2: Train Models**

Once data collection is complete, train each deliverable independently:

```bash
# Deliverable 1: MLP for position prediction
python deliverable1_mlp.py

# Deliverable 2: CNN for position prediction
python deliverable2_cnn.py

# Deliverable 3: Encoder-Decoder for image reconstruction
python deliverable3_encoder_decoder.py
```

Each script will:
* Load the dataset from `data/`
* Train the model for the specified number of epochs
* Save the trained weights to `models/`
* Generate loss curves and evaluation results in `assets/`

### Alternative: Download Pre-trained Models

If you prefer to skip training, pre-trained model weights are available:

## Model Weights
The trained model weights (`.pth` files) for all three deliverables are too large for GitHub and have been hosted externally. 
* **[Download Model Weights Here](https://drive.google.com/drive/folders/1vRdnbELEyx2eLgmHo1ZS9KkYHVwp4oMY?usp=drive_link)** 
* Download the files and place them inside a `models/` directory at the root of this project before running any evaluation scripts.

---

## Data Collection & Dataset Structure

### Data Collection Pipeline (`data_collector.py`)

The training data is generated through physics-based simulation using MuJoCo. The data collection process runs 4 parallel processes, each collecting 250 samples, for a total of **1,000 training examples**.

**Collection Process:**
1. For each sample, a random object (box or sphere) is spawned at a random position on the table
2. The initial scene is captured as `img_before` (128×128 RGB image)
3. A random pushing action (0-3) is selected and executed by the robotic arm
4. The final scene is captured as `img_after` with the object's new position recorded
5. The environment is reset for the next sample

**Data Format:**
Each sample contains:
* **`imgs_before`**: Initial RGB image (3×128×128) before the push action
* **`actions`**: One of 4 discrete pushing actions (0: Back→Front, 1: Front→Back, 2: Left→Right, 3: Right→Left)
* **`positions`**: Final (x, y) coordinates of the object after the push
* **`imgs_after`**: Final RGB image (3×128×128) after the push action

The data is saved in 4 parts (`*_0.pt`, `*_1.pt`, `*_2.pt`, `*_3.pt`) for efficient parallel collection.

### Dataset Class (`dataset.py`)

The `RobotControlDataset` class handles loading and preprocessing:
* **Loads** all 4 data parts and concatenates them into unified tensors
* **One-hot encodes** the discrete actions into 4-dimensional vectors
* **Normalizes** RGB images from `[0, 255]` to `[0.0, 1.0]` range for neural network training
* **Returns** tuples of `(img_before, action, position, img_after)` for flexible training across all three deliverables

**Dataset Statistics:**
* Total samples: 1,000
* Train/Test split: 800/200 (80%/20%)
* Image dimensions: 3×128×128 (RGB)
* Action space: 4 discrete directions (one-hot encoded)
* Position space: 2D continuous coordinates (x, y)

---

## Deliverable 1: Object Position Prediction (MLP)

**Objective:** Predict the final $(x, y)$ coordinates of an object from initial image and action using Multi-Layer Perceptron (MLP).

**Architecture:**
* The input consists of a $128 \times 128$ RGB image (`img_before`) and a 4-dimensional one-hot encoded action vector.
* The image is flattened into a 1D vector of $49,152$ features and concatenated with the action vector.
* The combined vector is passed through a dense network with hidden layers of sizes `[256, 64]`, utilizing ReLU activations and Dropout (0.2) for regularization.
* The output is a 2D coordinate representing the object's final position.

**Results:**
* **Final Test Error (MSE):** `0.058539`
![MLP Loss Curve](assets/mlp_loss_curve.png)

---

## Deliverable 2: Object Position Prediction (Two-Stream CNN)

**Objective:** Predict the final $(x, y)$ coordinates of an object from initial image and action using a Convolutional Neural Network (CNN) to better preserve spatial hierarchies.

**Architecture:**
* **Visual Stream:** A 4-layer CNN (channels: 16 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 64) with $3 \times 3$ kernels and $2 \times 2$ Max Pooling compresses the $128 \times 128$ image into a compact spatial feature map.
* **Fusion Stream:** The visual features are flattened, concatenated with the 4D action vector, and passed through a dense regression head (`[256, 64]`) to output the final 2D coordinates.

**Results:**
* The CNN successfully reduced the computational overhead of the massive linear layers in the MLP while improving feature extraction through spatial invariance.
* **Final Test Error (MSE):** `0.001951`
![CNN Loss Curve](assets/cnn_loss_curve.png)

---

## Deliverable 3: Post-Action Image Reconstruction (Encoder-Decoder CNN)

**Objective:** Generate the complete final $128 \times 128$ RGB image (`img_after`) based on the initial visual state and the applied action.

**Architecture & Engineering Choices:**
* **Bottleneck Design:** The encoder compresses the image down to a $32 \times 32$ spatial grid with 256 channels. Unlike more aggressive compression (e.g., $16 \times 16$), I think stopping at $32 \times 32$ preserves the physical footprint of the small target object, preventing it from being mathematically erased by the convolutional filters.
* **Action Broadcasting:** The 4D action vector is passed through a dense embedding layer, expanded, and spatially concatenated to the $32 \times 32$ visual bottleneck.
* **Bilinear Upsampling:** To prevent the checkerboard artifacts common in generative transposed convolutions, the decoder relies on Bilinear Upsampling followed by standard `Conv2d` layers.
* **Combined Loss Function:** Due to severe pixel class imbalance (the moving object constitutes a tiny fraction of the image), I implemented a custom loss function. It combines `MSELoss` (to stabilize the static background and robotic arm) with a weighted `L1Loss` targeting the Red channel ($\times 2.0$) to act as a soft prior, gently forcing the network's attention toward the object's movement.

**Analysis of Results:**
The network successfully learns the static background, the final retracted state of the robotic arm, and the directional movement of the object. However, the predicted objects exhibit visual blurring, sometimes seem duplicated and sometimes not reconstructed. I think this is likely due to the inherent difficulty of the task and the pixel-level precision required. The combined loss function I implemented helps mitigate this by emphasizing the importance of accurately reconstructing the red channel, which is crucial for identifying the object's position and movement. The final test error of `0.005313` indicates that while the model captures the overall structure and movement, there is still room for improvement in achieving sharper reconstructions.

**Results:**
* **Final Test Error (Combined Loss: MSE + L1):** `0.005313`

### Loss Curve
![Encoder-Decoder Loss Curve](assets/encoder_decoder_loss_curve.png)

### Reconstruction Grid
*(Left: Initial State & Action, Middle: Ground Truth Final State, Right: Predicted Final State. Each row corresponds to a separate simulation and is from the test set -not seen during training-)*
![Reconstruction Comparison](assets/reconstruction_comparison.png)