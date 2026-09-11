"""
Handwritten Alphabet Recognition using an Artificial Neural Network (ANN)
-------------------------------------------------------------------------
Streamlit app that trains a from-scratch NumPy ANN on REAL handwritten letters
(the EMNIST Letters dataset from NIST: 145,600 handwritten A-Z samples) and
predicts the letter in an image you draw in Paint and upload.

No TensorFlow / Keras / scikit-learn: the network (forward pass,
backpropagation, Adam optimizer) is pure NumPy, so it runs on any Python
version, including 3.14.

Dataset: EMNIST Letters (Cohen et al., 2017). Put these files in data/emnist/
next to this script:
    emnist-letters-train-images-idx3-ubyte.gz
    emnist-letters-train-labels-idx1-ubyte.gz
    emnist-letters-test-images-idx3-ubyte.gz
    emnist-letters-test-labels-idx1-ubyte.gz
EMNIST merges upper and lower case, so each of the 26 classes covers both
'A' and 'a', 'B' and 'b', and so on.

Run with:
    streamlit run handwritten_alphabet_ann.py

Required packages:
    pip install streamlit numpy pandas pillow
"""

import gzip
import os
import string
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
LETTERS = list(string.ascii_uppercase)   # A-Z
IMG_SIZE = 28                            # EMNIST images are 28x28
N_INPUTS = IMG_SIZE * IMG_SIZE           # 784 flattened pixels

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data", "emnist")
MODEL_PATH = os.path.join(APP_DIR, "models", "emnist_letters_ann.npz")

# Geometry of an EMNIST letter: it fills a 24x24 box centred in the 28x28
# image (2px border), and its strokes cover about 16% of the pixels.
# Uploaded drawings are normalised to the same geometry before prediction.
LETTER_BOX = 24
TARGET_INK = 0.16

HIDDEN_OPTIONS = {"256 → 128": [256, 128], "512 → 256": [512, 256], "128 → 128": [128, 128]}

st.set_page_config(page_title="Handwritten Alphabet ANN", page_icon="✍️", layout="centered")


# --------------------------------------------------------------------------------
# EMNIST loading (idx format, gzip-compressed)
# --------------------------------------------------------------------------------
def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def emnist_files(data_dir):
    return {
        f"{split}_{kind}": os.path.join(data_dir, f"emnist-letters-{split}-{kind}-idx{3 if kind == 'images' else 1}-ubyte.gz")
        for split in ("train", "test") for kind in ("images", "labels")
    }


# cache_resource (not cache_data) so the ~400 MB of arrays is shared, not copied every rerun
@st.cache_resource(show_spinner=False)
def load_emnist_letters(data_dir):
    files = emnist_files(data_dir)

    def split(name):
        images = load_idx(files[f"{name}_images"])
        labels = load_idx(files[f"{name}_labels"])
        # EMNIST stores every image transposed; labels run 1..26 (1 = A/a).
        X = images.transpose(0, 2, 1).reshape(len(images), -1).astype(np.float32) / 255.0
        return X, labels.astype(np.int64) - 1

    X_train, y_train = split("train")
    X_test, y_test = split("test")
    return X_train, y_train, X_test, y_test


def train_val_split(X, y, val_fraction=0.05, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_fraction)
    return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]


# --------------------------------------------------------------------------------
# Pure NumPy Artificial Neural Network (feedforward, ReLU, softmax, Adam)
# --------------------------------------------------------------------------------
class NumpyANN:
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)  # He initialization (good for ReLU)
            self.weights.append(rng.normal(0, std, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

        # Adam optimizer state
        self.mW = [np.zeros_like(w) for w in self.weights]
        self.vW = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        zs = []
        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.relu(Z)
            zs.append(Z)
            activations.append(A)
        Z = A @ self.weights[-1] + self.biases[-1]
        A = self.softmax(Z)
        zs.append(Z)
        activations.append(A)
        return activations, zs

    def compute_loss_acc(self, X, y):
        probs = self.predict_proba(X)
        n = X.shape[0]
        loss = float(np.mean(-np.log(probs[np.arange(n), y] + 1e-9)))
        acc = float(np.mean(np.argmax(probs, axis=1) == y))
        return loss, acc

    def train_batch(self, X, y, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                    label_smoothing=0.1, weight_decay=1e-4):
        n = X.shape[0]
        num_classes = self.layer_sizes[-1]
        activations, zs = self.forward(X)
        probs = activations[-1]

        # Label smoothing: soft targets stop the softmax from becoming
        # "100% confident and still wrong".
        Y_smooth = np.full_like(probs, label_smoothing / num_classes)
        Y_smooth[np.arange(n), y] += 1 - label_smoothing

        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        dZ = (probs - Y_smooth) / n
        grads_W[-1] = activations[-2].T @ dZ
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ self.weights[-1].T
        for i in reversed(range(len(self.weights) - 1)):
            dZ = dA_prev * self.relu_deriv(zs[i])
            grads_W[i] = activations[i].T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dZ @ self.weights[i].T

        self.t += 1
        for i in range(len(self.weights)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (grads_W[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            # Decoupled weight decay (AdamW-style) keeps weights small.
            self.weights[i] -= lr * (mW_hat / (np.sqrt(vW_hat) + eps) + weight_decay * self.weights[i])

            # Momentum for pixels that are almost never inked (image borders)
            # decays into the float32 denormal range, where CPU arithmetic is
            # many times slower. Flushing those values to 0 changes nothing.
            np.putmask(self.mW[i], np.abs(self.mW[i]) < 1e-30, 0.0)
            np.putmask(self.vW[i], self.vW[i] < 1e-30, 0.0)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (grads_b[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def predict_proba(self, X, batch_size=4096):
        return np.vstack([self.forward(X[s:s + batch_size])[0][-1]
                          for s in range(0, len(X), batch_size)])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ---- save / load, so the app does not have to retrain on every start ----
    def save(self, path, meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        arrays.update({f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        np.savez_compressed(path, layer_sizes=np.asarray(self.layer_sizes), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            model = cls([int(s) for s in f["layer_sizes"]])
            model.weights = [f[f"W{i}"] for i in range(len(model.weights))]
            model.biases = [f[f"b{i}"] for i in range(len(model.biases))]
            meta = {k[5:]: (f[k].item() if f[k].ndim == 0 else f[k].tolist())
                    for k in f.files if k.startswith("meta_")}
        return model, meta


def fit_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128,
              lr=0.001, seed=0, on_epoch=None):
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    train_probe = rng.choice(n, size=min(n, 10000), replace=False)  # cheap train-accuracy estimate
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            model.train_batch(X_train[batch], y_train[batch], lr=lr)

        train_loss, train_acc = model.compute_loss_acc(X_train[train_probe], y_train[train_probe])
        val_loss, val_acc = model.compute_loss_acc(X_val, y_val)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if on_epoch:
            on_epoch(epoch + 1, epochs, train_acc, val_acc)

    return history


# --------------------------------------------------------------------------------
# Uploaded image -> 28x28 array that looks like an EMNIST letter
# --------------------------------------------------------------------------------
BACKGROUND_OPTIONS = [
    "Auto-detect",
    "Light background, dark letter (Paint default)",
    "Dark background, light letter",
]


def flatten_alpha(img):
    # A transparent PNG background is paper, not black ink: paste onto white.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return img.convert("RGB")


def to_ink(img, background=BACKGROUND_OPTIONS[0]):
    """Grayscale float array where ink = 1 and paper = 0."""
    arr = np.asarray(flatten_alpha(img).convert("L"), dtype=np.float32) / 255.0
    if background == BACKGROUND_OPTIONS[0]:
        # The image border is almost always paper, so a bright border
        # means dark ink on light paper.
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        invert = np.median(border) > 0.5
    else:
        invert = background == BACKGROUND_OPTIONS[1]
    return 1.0 - arr if invert else arr


def _to_pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _crop_to_ink(a, threshold=0.2):
    ys, xs = np.where(a > threshold)
    if len(xs) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _fit_in_box(a):
    """Scale the letter so its longer side is LETTER_BOX px, centred in 28x28."""
    h, w = a.shape
    scale = LETTER_BOX / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(_to_pil(a).resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0
    out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y0, x0 = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def preprocess_upload(img, background=BACKGROUND_OPTIONS[0]):
    ink = to_ink(img, background)

    # Stretch contrast so faint pencil/pen strokes reach full intensity.
    lo, hi = ink.min(), ink.max()
    if hi - lo < 1e-6:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ink = (ink - lo) / (hi - lo)

    letter = _crop_to_ink(ink)
    if letter is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    letter = np.where(letter > 0.2, letter, 0.0)

    # Work at ~96px (EMNIST was made from 128px scans) so stroke thickening
    # below is fast and independent of the uploaded image's resolution.
    h, w = letter.shape
    scale = 96 / max(h, w)
    letter = np.asarray(_to_pil(letter).resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                               Image.LANCZOS), dtype=np.float32) / 255.0

    # EMNIST strokes are thick. A thin Paint brush would all but vanish at
    # 28x28, so thicken the strokes step by step until the letter carries
    # about as much ink as a typical EMNIST letter.
    padded = np.pad(letter, 8)
    result = None
    for size in (1, 3, 5, 7, 9, 11, 13, 15):
        thick = padded if size == 1 else np.asarray(
            _to_pil(padded).filter(ImageFilter.MaxFilter(size)), dtype=np.float32) / 255.0
        result = _fit_in_box(_crop_to_ink(thick))
        if np.mean(result > 0.5) >= TARGET_INK:
            break
    return result


# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("✍️ Handwritten Alphabet Recognition with an ANN")
st.write(
    "A feedforward Artificial Neural Network, written from scratch in NumPy, "
    "trained on **real handwritten letters** from the EMNIST Letters dataset "
    "(NIST). Draw a letter in Paint, upload it, and the network predicts it."
)

files = emnist_files(DATA_DIR)
missing = [os.path.basename(p) for p in files.values() if not os.path.exists(p)]
if missing:
    st.error(
        "EMNIST Letters files not found in `" + DATA_DIR + "`. Missing: "
        + ", ".join(missing)
        + ". Download the EMNIST dataset (gzip version) from "
        "https://www.nist.gov/itl/products-and-services/emnist-dataset and copy the "
        "four `emnist-letters-*` files into that folder."
    )
    st.stop()

with st.spinner("Loading EMNIST Letters..."):
    X_all, y_all, X_test, y_test = load_emnist_letters(DATA_DIR)

if "model" not in st.session_state:
    st.session_state.model, st.session_state.meta = None, None
    if os.path.exists(MODEL_PATH):
        st.session_state.model, st.session_state.meta = NumpyANN.load(MODEL_PATH)

# ---- Sidebar: training controls ----
st.sidebar.header("⚙️ Training Settings")
n_train = st.sidebar.select_slider(
    "Training images", options=[20000, 40000, 60000, 80000, 100000, len(X_all)], value=len(X_all)
)
hidden_label = st.sidebar.selectbox("Hidden layers", list(HIDDEN_OPTIONS))
epochs = st.sidebar.slider("Epochs", 1, 30, 10)
learning_rate = st.sidebar.select_slider(
    "Learning rate", options=[0.0003, 0.0005, 0.001, 0.002, 0.003], value=0.001
)
train_button = st.sidebar.button("🚀 Train Model")
st.sidebar.caption(
    "Training on all 124,800 images takes about 10 s per epoch. "
    "The trained model is saved to disk and loaded automatically next time."
)

if train_button:
    X_train, y_train, X_val, y_val = train_val_split(X_all, y_all)
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    layers = [N_INPUTS] + HIDDEN_OPTIONS[hidden_label] + [len(LETTERS)]
    model = NumpyANN(layer_sizes=layers)

    progress = st.progress(0.0, text="Training ANN...")
    started = time.time()

    def on_epoch(epoch, total, train_acc, val_acc):
        progress.progress(epoch / total, text=f"Epoch {epoch}/{total} — train acc {train_acc:.1%}, "
                                             f"validation acc {val_acc:.1%}")

    history = fit_model(model, X_train, y_train, X_val, y_val,
                        epochs=epochs, lr=learning_rate, on_epoch=on_epoch)
    _, test_acc = model.compute_loss_acc(X_test, y_test)
    meta = {
        "test_accuracy": test_acc, "epochs": epochs, "train_images": len(X_train),
        "learning_rate": learning_rate, "hidden": hidden_label,
        "train_seconds": time.time() - started,
        "accuracy": history["accuracy"], "val_accuracy": history["val_accuracy"],
    }
    model.save(MODEL_PATH, meta)
    st.session_state.model, st.session_state.meta = model, meta
    progress.empty()
    st.success(f"Training complete in {meta['train_seconds']:.0f}s. "
               f"Accuracy on the 20,800 unseen EMNIST test letters: {test_acc:.2%}")

with st.expander("👀 Peek at the training data (real handwriting)"):
    rng = np.random.default_rng(0)
    rows = [np.hstack([X_all[i].reshape(IMG_SIZE, IMG_SIZE)
                       for i in rng.choice(np.where(y_all == c)[0], 10, replace=False)])
            for c in range(len(LETTERS))]
    st.image(_to_pil(np.vstack(rows)).resize((10 * IMG_SIZE * 2, 26 * IMG_SIZE * 2), Image.NEAREST),
             caption="10 random training samples per letter (A to Z, top to bottom). "
                     "Upper and lower case share a class.")

model = st.session_state.model
meta = st.session_state.meta

# ---- Model performance ----
if model is not None:
    st.subheader("📈 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{meta['test_accuracy']:.2%}")
    c2.metric("Hidden layers", meta["hidden"])
    c3.metric("Trained on", f"{meta['train_images']:,} images, {meta['epochs']} epochs")

    st.line_chart(pd.DataFrame({"train accuracy": meta["accuracy"],
                                "validation accuracy": meta["val_accuracy"]},
                               index=range(1, len(meta["accuracy"]) + 1)))

    with st.expander("Per-letter accuracy and most common mix-ups"):
        pred_test = model.predict(X_test)
        per_letter = [float(np.mean(pred_test[y_test == c] == c)) for c in range(len(LETTERS))]
        st.bar_chart(pd.DataFrame({"accuracy": per_letter}, index=LETTERS))
        confusion = np.zeros((len(LETTERS), len(LETTERS)), dtype=int)
        np.add.at(confusion, (y_test, pred_test), 1)
        np.fill_diagonal(confusion, 0)
        top = np.dstack(np.unravel_index(np.argsort(confusion, axis=None)[::-1][:8], confusion.shape))[0]
        st.table(pd.DataFrame(
            [{"true letter": LETTERS[t], "predicted as": LETTERS[p], "count (of 800)": int(confusion[t, p])}
             for t, p in top]))

st.divider()

# ---- Input: upload images of hand-drawn letters (e.g. from Paint) ----
st.subheader("✏️ Upload Handwritten Letters")

if model is None:
    st.info("Train the model first using the sidebar button.")
else:
    st.markdown(
        "**Tips for drawing in Paint:**\n"
        "- One letter per image, drawn large; any canvas size works.\n"
        "- Capital or small letters both work (EMNIST merges them).\n"
        "- Brush thickness does not matter much: strokes are normalised "
        "to match the training data.\n"
        "- Black on white (Paint's default) is detected automatically."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more letter images", type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )
    background = st.selectbox("Image colours", BACKGROUND_OPTIONS)

    for uploaded_file in uploaded_files or []:
        img = Image.open(uploaded_file)
        arr = preprocess_upload(img, background)
        probs = model.predict_proba(arr.reshape(1, -1))[0]
        top3 = np.argsort(probs)[::-1][:3]

        st.markdown(f"#### {uploaded_file.name}")
        col1, col2, col3 = st.columns([1, 1, 1.4])
        with col1:
            st.image(flatten_alpha(img), caption="Your image", width=140)
        with col2:
            st.image(_to_pil(arr).resize((140, 140), Image.NEAREST),
                     caption="What the ANN sees (28x28)")
        with col3:
            st.markdown(f"### Predicted: **{LETTERS[top3[0]]}**")
            st.write(f"{probs[top3[0]]:.1%} confidence")
            st.caption("Next best: " + ", ".join(f"{LETTERS[i]} ({probs[i]:.1%})" for i in top3[1:]))
        if len(uploaded_files) == 1:
            st.bar_chart(pd.DataFrame({"Probability": probs}, index=LETTERS))

    if uploaded_files:
        st.caption(
            "⚠️ If 'What the ANN sees' looks blank or inverted, set 'Image colours' "
            "by hand instead of 'Auto-detect'."
        )

st.divider()
st.caption(
    "Trained on EMNIST Letters (Cohen et al., 2017, NIST): 124,800 training and "
    "20,800 test images of handwritten letters. A plain ANN (multilayer perceptron) "
    "reaches roughly 90% on this test set; letters that look alike when handwritten "
    "(I/L, G/Q, U/V) cause most of the errors."
)
