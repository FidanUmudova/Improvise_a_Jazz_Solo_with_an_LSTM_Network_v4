# 🎷 Improvise a Jazz Solo with an LSTM Network

This project is a deep learning application built using Keras and TensorFlow to generate improvised jazz solos using Recurrent Neural Networks (RNNs) and LSTM (Long Short-Term Memory) architectures. It covers sequential data modeling, time-step iterations, and inference generation.

---

## 🚀 Core Components & Functions

The project implements three main functions corresponding to different parts of the sequence generation pipeline:

### 1. `djmodel` (Training Model)
* **Description:** Constructed using the Keras Functional API to handle music sequences of length $T_x$.
* **Workflow:** At each time step, the input $X_t$ passes through a `reshaper` layer and is fed into an LSTM cell. The resulting hidden state is then propagated through a dense + softmax layer to predict the next note.

### 2. `music_inference_model` (Inference Model)
* **Description:** Designed to generate a new, original sequence of music of length $T_y$ using the trained weights.
* **Workflow:** In a loop, the highest-probability note predicted at the current step is converted into a one-hot vector and fed as the input for the next time step.

### 3. `predict_and_sample` (Prediction & Sampling)
* **Description:** A helper function that executes the inference model to generate a new melody sequence from initial inputs.
* **Workflow:** 
  1. Runs the inference model to obtain the prediction list (`pred`).
  2. Extracts the most likely indices using `np.argmax`.
  3. Converts these indices back into `one-hot` vector representations using `to_categorical`.

---

## 🛠️ Technologies Used
* **Python**
* **TensorFlow / Keras** (Functional API, LSTM, Dense, RepeatVector, Input)
* **NumPy**

---
