# ✨Word Embeddings using CBOW and Skip-gram Models

A project implementing word embeddings using CBOW and Skip-gram models to generate vector representations of words and capture semantic relationships.

This repository contains my implementation of **Word Embeddings** using two popular models: **Continuous Bag of Words (CBOW)** and **Skip-gram**. The project demonstrates how these models work to generate vector representations of words, capturing semantic relationships between them. Word embeddings are a crucial concept in Natural Language Processing (NLP) and are widely used for various NLP tasks such as sentiment analysis, machine translation, and text classification.

## Key Features

- **CBOW Model**: Predicts a target word from the surrounding context words. It aims to maximize the probability of the target word given its context.
- **Skip-gram Model**: The reverse of CBOW, it predicts the context words given a target word. Skip-gram works well for handling rare words.

## Project Overview

In this project, I used the CBOW and Skip-gram models to create word embeddings. The word embeddings are generated using a neural network-based approach, and both models are trained on a sample corpus. After training, the word vectors are visualized to analyze how similar words cluster in a vector space.

### Workflow

1. **Data Preprocessing**:
   - A sample text corpus is used for training.
   - Tokenization and cleaning of the text data to prepare it for the embedding models.
   
2. **Model Implementation**:
   - **CBOW Model**: Uses context words (surrounding words) to predict the target word.
   - **Skip-gram Model**: Uses a target word to predict surrounding context words.

3. **Training**:
   - The models are trained using the processed corpus to generate word embeddings.
   
4. **Visualization**:
   - Visualizing the generated word embeddings using techniques like t-SNE or PCA to plot the word vectors in 2D/3D space.

5. **Evaluation**:
   - Analyzing the quality of the embeddings by checking the relationships between words in the vector space (e.g., similarity between synonyms and antonyms).

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Aliharis007/Word-Embeddings.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Word-Embeddings
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

   You can also install individual libraries if you prefer:

   ```bash
   pip install numpy pandas matplotlib tensorflow
   ```

## Files and Folders

- **`CBOW_img.png`**: Visualization of the CBOW model training results.
- **`REPORT.pdf`**: Detailed report describing the methodology and results of the project.
- **`Word Embeddings.pdf`**: Documentation of the project and model architectures.
- **`prediction.png`**: Example of word vector predictions.
- **`skip_gram_img.png`**: Visualization of the Skip-gram model training results.
- **`text_corpus.txt`**: Sample corpus used for training the models.

## 🚀Usage

### Step 1: Data Preprocessing

- The `text_corpus.txt` file contains a sample corpus. You can replace this with any custom dataset for training.

### Step 2: Train CBOW and Skip-gram Models

- Run the script to train the CBOW and Skip-gram models using the `train.py` file (or whichever script you use for training).

   ```bash
   python train.py
   ```

- The models will generate word embeddings and save them to files for further use.

### Step 3: Visualize Word Embeddings

- After training, you can visualize the generated embeddings using the `visualize.py` script:

   ```bash
   python visualize.py
   ```

- This will generate 2D/3D plots of the word vectors, showing how similar words cluster together.

## Example Results

Here are some example results based on the word embeddings:

- **CBOW**: Predicted word relationships using context words.
- **Skip-gram**: Predicted context words based on a target word.
