## 🧐 Deep Learning Assignment 2

**Course:** Deep Learning - 22CAC04\
**Institution:** Chaitanya Bharathi Institute of Technology\
**Department:** Information Technology\
**Due Date:** 20-04-25

### 🔍 Overview

This repository contains the implementation of **Question 1** and **Question 2** of the Deep Learning Assignment 2.

---

## 📌 Question 1: Latin-to-Devanagari Transliteration

### 🚀 Objective

To build a flexible RNN-based seq2seq architecture for transliterating Latin script inputs to their corresponding Devanagari script representations. The model supports multiple cell types: **SimpleRNN**, **GRU**, and **LSTM**, with tunable hyperparameters.

---

### 🗂️ Dataset

Dataset used: [Dakshina Dataset (Google)](https://github.com/google-research-datasets/dakshina)\
Files used:

- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Each file contains columns:

- Devanagari script
- Latin transliteration
- Frequency count

---

### 🧱 Model Architecture

1. **Embedding Layer** for both encoder and decoder
2. **Encoder RNN (LSTM / GRU / SimpleRNN)** - processes the Latin script input
3. **Decoder RNN (LSTM / GRU / SimpleRNN)** - generates the Devanagari script character-by-character using the final encoder state
4. **Dense Layer** with softmax activation for character prediction

**Flexibility:**

- Embedding Dimension
- Hidden Units
- RNN Cell Type (`'lstm'`, `'gru'`, `'rnn'`)
- Number of Layers (extendable in the function)

---

### 🧮 Theoretical Analysis

#### a) Total Number of Computations

Let:

- `m` = embedding dimension
- `k` = hidden size
- `T` = sequence length
- `V` = vocabulary size

Total computations (approx):\
Encoder: O(T × (m×k + k²))\
Decoder: O(T × (m×k + k² + k×V))

#### b) Total Number of Parameters

Encoder LSTM: 4 × (k×(k + m + 1))\
Decoder LSTM: 4 × (k×(k + m + 1))\
Dense Output: k × V\
Embedding Layers: V × m (each for encoder and decoder)

---

### 📊 Training Details

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Batch Size:** 64
- **Epochs:** 30
- **Validation Accuracy:** \~94.6%
- **Test Accuracy:** **0.9457**

---

### 📈 Sample Predictions

| Input (Latin) | Target (Devanagari) | Predicted |
| ------------- | ------------------- | --------- |
| a n k         | अ ं क               | ऐंक       |
| a n k a       | अ ं क               | अंका      |
| a n k i t     | अ ं क ि त           | अंकित     |
| a n a k o n   | अ ं क ो ं           | अनकों     |
| a n k h o n   | अ ं क ो ं           | अंखों     |

---

### 🧲 Evaluation

```bash
Test Accuracy: 0.9457
```

---

### 🛠️ How to Run

#### 🔧 Install Requirements

```bash
pip install tensorflow==2.12.0 pandas gdown
```

#### ▶️ Run Training

Ensure the `.tsv` files from Dakshina dataset are in your working directory.

```python
python main_seq2seq_transliteration.py
```

---

### 📂 File Structure

```
.
├── README.md
├── main_seq2seq_transliteration.py  # All code for Q1
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```

---

📘 References

- [Keras LSTM Seq2Seq Example](https://keras.io/examples/nlp/lstm_seq2seq/)
- [Machine Learning Mastery - Seq2Seq](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)

---

## 🎶 Question 2: GPT-2 Fine-Tuning for Lyric Generation

### 🎯 Objective

To fine-tune a pre-trained GPT-2 language model on a dataset of English poetry/lyrics, enabling it to generate new song-like text sequences.

---

### 📁 Dataset

Used: [Paul Timothy Mooney's Poetry Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/poetry)

- The `.txt` files were combined and cleaned.
- Each poem was appended with `<|endoftext|>` to mark the end of a sample.

---

### 🧪 Tokenization and Preprocessing

- Used `GPT2Tokenizer` with `eos_token` as pad token.
- Applied max length padding (`512`) and truncation.
- Prepared HuggingFace `Dataset` for training.
- Created `input_ids`, `attention_mask`, and `labels` for causal language modeling.

---

### 🧠 Model and Training Configuration

- Model: `GPT2LMHeadModel` (pretrained GPT-2)
- Trainer used from HuggingFace Transformers

**Training Arguments:**

- `epochs`: 5
- `batch size`: 4
- `learning_rate`: 5e-5
- `output_dir`: `./lyrics_generator`
- Logging & saving every 500 steps

---

### 📉 Training Output

```
Training Loss: 2.79
Train Runtime: ~41 seconds for 5 epochs
```

Model saved to: `./fine_tuned_lyrics_gpt2`

---

### 🎤 Sample Output

**Prompt:** `When the night comes`

**Generated Lyrics:**

```
When the night comes
And the moon rises in it, to see us
The dream that makes us fall
And all that's to be forgotten
The dream is coming true.

And you, like me, have told me all
The tale of all this
And you had this one word to tell of it all
And this word is still the same
And what do I know how to tell it all
So tell me what is the true meaning of this dream
```

---

### 🛠️ How to Run

#### 🔧 Install Requirements

```bash
pip install transformers datasets kagglehub
```

#### ▶️ Fine-Tune Model

```python
python gpt2_lyrics_finetune.py
```

---

### 📂 File Structure

```
.
├── README.md
├── gpt2_lyrics_finetune.py
└── /fine_tuned_lyrics_gpt2/  # Saved model
```

---

📘 References

- [GPT-2 Fine-tuning Tutorial](https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a)
- [Poetry Dataset on Kaggle](https://www.kaggle.com/paultimothymooney/poetry)

---

👤 **Author:** Eega Saikumar

