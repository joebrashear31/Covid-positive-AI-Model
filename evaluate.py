import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# CONFIG
# =========================
dataset_dir = "dataset"          # expects dataset/covid and dataset/normal
img_size = (128, 128)
batch_size = 32
model_path = "covid_xray_model.h5"
threshold = 0.5
show_gradcam = True

# =========================
# DATA: match your training split
# =========================
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class indices:", val_gen.class_indices)
target_names = [k for k, _ in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]

# =========================
# LOAD MODEL
# =========================
model = load_model(model_path)
print(f"Loaded model from: {model_path}")

# Optional: force-construct internal graph (safe across TF/Keras versions)
_ = model.predict(np.zeros((1, *img_size, 3), dtype=np.float32))

# =========================
# PREDICT
# =========================
steps = math.ceil(val_gen.samples / val_gen.batch_size)
y_prob = model.predict(val_gen, steps=steps, verbose=1).ravel()
y_true = val_gen.classes
y_pred = (y_prob >= threshold).astype(int)

# =========================
# SUMMARY METRICS
# =========================
print("\nClassification Report (threshold = {:.2f}):".format(threshold))
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# =========================
# PLOTS
# =========================
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel="True label", xlabel="Predicted label")

    fmt = ".2f" if normalize else "d"
    thresh_val = np.nanmax(cm) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if (not np.isnan(val) and val > thresh_val) else "black")
    fig.tight_layout()
    plt.show()

plot_confusion_matrix(cm, classes=target_names, normalize=False, title="Confusion Matrix (Counts)")
plot_confusion_matrix(cm, classes=target_names, normalize=True, title="Confusion Matrix (Normalized)")

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()

# =========================
# VISUALIZE SAMPLE PREDICTIONS
# =========================
def show_samples(indices, title):
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for i, idx in enumerate(indices[:rows*cols]):
        img_path = val_gen.filepaths[idx]
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        true_label = target_names[y_true[idx]]
        pred_label = target_names[y_pred[idx]]
        prob = y_prob[idx]
        correct = (y_true[idx] == y_pred[idx])
        plt.title(f"T:{true_label}\nP:{pred_label} ({prob:.2f})",
                  color=("green" if correct else "red"))
        plt.axis("off")
    plt.suptitle(title, y=0.98, fontsize=14)
    plt.tight_layout()
    plt.show()

rng = np.random.default_rng(42)
all_idx = np.arange(len(y_true))
rng.shuffle(all_idx)
show_samples(all_idx[:16], title="Random Validation Samples")

mis_idx = np.where(y_true != y_pred)[0]
if len(mis_idx) > 0:
    show_samples(mis_idx[:16], title="Misclassified Samples")
else:
    print("No misclassified samples to display.")

# =========================
# OPTIONAL: Grad-CAM (robust functional rebuild)
# =========================
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def build_grad_model_from_layers(model, last_conv_name, input_shape):
    """
    Rebuilds a functional graph using the SAME layer instances (weights shared),
    returns a model that outputs both the last conv feature map and final predictions.
    Avoids 'sequential has never been called' issues.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_name:
            conv_out = x
    preds = x
    return tf.keras.Model(inputs=inputs, outputs=[conv_out, preds])

def make_gradcam_heatmap(img_array, model, last_conv_name, input_shape):
    grad_model = build_grad_model_from_layers(model, last_conv_name, input_shape)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        loss = preds[:, 0]  # binary sigmoid output neuron
    grads = tape.gradient(loss, conv_outputs)             # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_outputs = conv_outputs[0]                        # (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (H, W)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def superimpose(img_path, heatmap, alpha=0.35):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    hm = tf.image.resize(heatmap[..., np.newaxis], (img.shape[0], img.shape[1])).numpy().squeeze()
    hm_uint8 = np.uint8(255 * hm)
    cmap = plt.get_cmap("jet")
    colored = cmap(hm_uint8)[..., :3] * 255  # drop alpha
    overlay = (colored * alpha + img).astype(np.uint8)
    return img.astype(np.uint8), overlay.astype(np.uint8)

if show_gradcam:
    last_conv_name = get_last_conv_layer_name(model)
    if last_conv_name is None:
        print("No Conv2D layer found for Grad-CAM.")
    else:
        print(f"Using last conv layer for Grad-CAM: {last_conv_name}")
        input_shape = (img_size[0], img_size[1], 3)
        candidates = mis_idx.tolist() if len(mis_idx) > 0 else all_idx[:8].tolist()

        for idx in candidates[:8]:
            img_path = val_gen.filepaths[idx]
            # match training preprocessing (rescale=1/255)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            heatmap = make_gradcam_heatmap(arr, model, last_conv_name, input_shape)
            original, overlay = superimpose(img_path, heatmap)

            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1); plt.imshow(original.astype(np.uint8)); plt.axis("off")
            t_lab = target_names[y_true[idx]]; p_lab = target_names[y_pred[idx]]
            plt.title(f"Original\nT:{t_lab} | P:{p_lab} ({y_prob[idx]:.2f})")
            plt.subplot(1,2,2); plt.imshow(overlay.astype(np.uint8)); plt.axis("off")
            plt.title("Grad-CAM Overlay")
            plt.tight_layout(); plt.show()
