# RNN / LSTM / GRU from Scratch in NumPy

This repo contains minimal, **from-scratch NumPy implementations** of:

- Vanilla RNN
- LSTM
- GRU

with a small training script for sequence prediction so you can **compare behavior and gradients**.

## Structure

- `src/datasets.py` – toy sequence dataset generators
- `src/rnn_numpy.py` – vanilla RNN (forward + BPTT)
- `src/lstm_numpy.py` – LSTM cell and sequence model
- `src/gru_numpy.py` – GRU cell and sequence model
- `src/train_sequence.py` – simple training loop to train any of the above
- `notebooks/exploration.ipynb` – optional: experiments, plots

## Install

```bash
git clone <your-repo-url> rnn-numpy-playground
cd rnn-numpy-playground

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
