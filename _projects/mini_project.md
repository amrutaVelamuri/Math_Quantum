---
layout: page
title: Quantum vs. Classical ML for Parkinson's Detection
description: A Variational Quantum Classifier compared to classical SVM/NN on gait data
img: assets/img/12.jpg
importance: 1
category: work
---

## Research question

> **How does a Variational Quantum Classifier (VQC) compare to a classical SVM
> (and neural network) in Parkinson's disease detection from gait data, measured
> across accuracy, precision, recall, and F1-score?**

## Background

I've spent the last three years researching Parkinson's disease (PD). PD affects
movement, and one of its clearest signatures is in **gait**, the way a person
walks. For this mini project I added a **quantum twist** to that work: instead of
only using classical machine-learning models, I trained a **Variational Quantum
Classifier** and put it head-to-head with the classical models I usually rely on.

## Data & features

I used the [PhysioNet *Gait in Parkinson's Disease*](https://physionet.org/content/gaitpdb/1.0.0/)
database, which records vertical ground-reaction force from sensors under each
foot as people walk. From each recording I extracted **8 simple features** from
the total force signal (mean, standard deviation, median, 25th/75th percentiles,
per-foot means, and mean step-to-step change).

- **306 recordings**, 8 features each
- **~70% of recordings are PD** (this class imbalance matters; see Discussion)

## Methods

- Standardize features, then a stratified **75/25 train/test split**.
- **Quantum model (VQC):** [PennyLane](https://pennylane.ai/), 8 qubits,
  `AngleEmbedding` to load the features plus 3 `StronglyEntanglingLayers`, trained
  with Adam for 25 steps against ±1 targets (mean-squared error).
- **Classical baselines:** SVM, Logistic Regression (LR), and a small
  neural network (MLP).

```python
import numpy as np, pandas as pd, re, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "physionet.org/files/gaitpdb/1.0.0"

X, y = [], []
for fp in Path(DATA_DIR).rglob("*.txt"):
    m = re.match(r"^(Ga|Ju|Si)(Co|Pt)\d+_\d+$", fp.stem, re.I)
    if not m: continue
    try:
        d = pd.read_csv(fp, sep="\t", header=None).apply(pd.to_numeric, errors="coerce").fillna(0)
        if d.shape[1] < 19: continue
        total = d.iloc[:, 17].values + d.iloc[:, 18].values
        feats = [total.mean(), total.std(), np.median(total),
                 np.percentile(total, 25), np.percentile(total, 75),
                 d.iloc[:, 17].values.mean(), d.iloc[:, 18].values.mean(),
                 np.abs(np.diff(total)).mean()]
        X.append(feats); y.append(0 if m.group(2).lower() == "co" else 1)
    except: continue
X, y = np.array(X), np.array(y)
print(f"{len(y)} recordings, {X.shape[1]} features, {y.mean():.0%} PD")

X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

dev = qml.device("default.qubit", wires=8)
@qml.qnode(dev)
def circuit(x, w):
    qml.AngleEmbedding(x, wires=range(8))
    qml.StronglyEntanglingLayers(w, wires=range(8))
    return qml.expval(qml.PauliZ(0))

w = pnp.array(np.random.normal(0, 0.1, qml.StronglyEntanglingLayers.shape(3, 8)), requires_grad=True)
opt = qml.AdamOptimizer(0.05)
target = np.where(ytr == 1, 1.0, -1.0)
cost = lambda w: pnp.mean((pnp.stack([circuit(x, w) for x in Xtr]) - target) ** 2)
for _ in range(25):
    w = opt.step(cost, w)
vqc_pred = np.array([1 if circuit(x, w) > 0 else 0 for x in Xte])

def show(name, pred):
    print(f"{name:5s} acc={accuracy_score(yte,pred):.2f}  prec={precision_score(yte,pred,zero_division=0):.2f}  "
          f"rec={recall_score(yte,pred,zero_division=0):.2f}  f1={f1_score(yte,pred,zero_division=0):.2f}")

print()
show("VQC", vqc_pred)
for name, clf in [("SVM", SVC()), ("LR", LogisticRegression(max_iter=2000)), ("NN", MLPClassifier((16,), max_iter=800))]:
    clf.fit(Xtr, ytr)
    show(name, clf.predict(Xte))
```

## Results

`306 recordings, 8 features, 70% PD`

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| **VQC** (quantum) | 0.69 | 0.70 | 0.96 | 0.81 |
| SVM   | 0.70 | 0.70 | 1.00 | 0.82 |
| LR    | 0.70 | 0.71 | 0.96 | 0.82 |
| NN    | 0.71 | 0.71 | 1.00 | 0.83 |

## Discussion

The headline result: **the quantum classifier is competitive with the classical
models**, within about 1 to 2 points on every metric, using only 8 qubits and 25
training steps with almost no tuning. That's encouraging for such a small quantum
circuit.

## Limitations & future work

- **Handle the class imbalance:** use class weights or balanced sampling, and
  report **balanced accuracy** and **ROC-AUC** instead of raw accuracy.
- **Richer features:** temporal gait dynamics (stride time, swing/stance,
  variability) and per-sensor signals, plus feature selection.
- **Stronger quantum model:** more qubits or layers, more training iterations, and
  proper hyperparameter tuning; compare data-embedding strategies.
- **More robust evaluation:** k-fold cross-validation and a held-out subject-wise
  test set so results generalize to new patients.
