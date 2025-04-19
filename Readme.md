Hindi Transliteration using Seq2Seq Model

This project implements a character-level Sequence-to-Sequence (Seq2Seq) model for Hindi transliteration, converting Romanized Hindi (Latin script) into Devanagari script using TensorFlow/Keras.

📂 Dataset

Dataset consists of paired Hindi words in Latin and Devanagari scripts, along with frequency counts. It includes:

hi.translit.sampled.train.tsv

hi.translit.sampled.dev.tsv

hi.translit.sampled.test.tsv

Each line: <devanagari> <latin> <count>

🚀 Training the Model

1. Install Requirements

pip install tensorflow==2.12.0 pandas gdown

2. Preprocess Data

Load and clean data

Tokenize sequences at character-level

Pad to fixed length T = 20

Add start (\t) and end (\n) tokens for decoder input/output

3. Build the Model

model, encoder_model, decoder_model = build_seq2seq_model(
    input_vocab_size, target_vocab_size,
    embedding_dim=64, hidden_units=128, cell_type='lstm')

4. Compile and Train

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([...], epochs=30, batch_size=64, validation_data=...)

5. Evaluate the Model

test_loss, test_acc = model.evaluate([X_test, decoder_input_test], decoder_output_test)
print(f"Test Accuracy: {test_acc:.4f}")

6. Sample Prediction

Input (Latin): ankur
Target (Devanagari): अंकुर
Predicted (Devanagari): अंकुर

🔢 Network Computation and Parameters

Let:

m = embedding size

k = hidden size of encoder/decoder

T = input/output sequence length (same)

V = vocabulary size

(a) Total Number of Computations

Encoder LSTM: For each time step, cost = 4(mk + k^2 + k) → over T steps:

Encoder: T * 4(mk + k^2 + k)

Decoder LSTM: Similar cost:

Decoder: T * 4(mk + k^2 + k)

Dense Softmax Layer: For each output step:

Dense: T * (k * V)

Total Computations:

= T * [8mk + 8k^2 + 8k + kV]

(b) Total Number of Parameters

Embedding Layer (source and target): 2 * (V * m)

LSTM Encoder: 4 * (m * k + k * k + k)

LSTM Decoder: 4 * (m * k + k * k + k)

Dense Layer: k * V + V

Total Parameters:

= 2Vm + 8mk + 8k^2 + 8k + kV + V

📊 Results from Best Model

Model: LSTM-based Seq2Seq (Embedding=64, Hidden=128)

Test Accuracy: 96.74%

Sample Predictions

Input (Latin)

Target (Devanagari)

Predicted (Devanagari)

ankur

अंकुर

अंकुर

sneha

स्नेहा

स्नेहा

ramesh

रमेश

रमेश

📚 Summary

Developed and trained a character-level Seq2Seq transliteration model.

Achieved high accuracy on test set.

Performed detailed analysis of computation and parameter count.

Successfully predicts Devanagari transliteration for unseen Romanized Hindi words.

✍️ Author

Eega SaikumarFeel
free to fork, star, or contribute!