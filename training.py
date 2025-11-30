from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np

categories = ["dragon", "angel", "mermaid", "castle", "flying saucer"]
filenames = ["dragon", "angel", "mermaid", "castle", "flying_saucer"]

#load data
X, y = [], []
for idx, (cat, filename) in enumerate(zip(categories, filenames)):
    data = np.load(f"{filename}.npy")

    #random sampling; shuffle before taking samples
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    data = data[indices[:5000]]

    X.append(data)
    y.extend([idx] * len(data))
    print(f"Loaded {len(data)} samples for {cat}")

X = np.concatenate(X)
y = np.array(y)

#check balance
unique, counts = np.unique(y, return_counts=True)
print("\nClass balance:")
for i, count in zip(unique, counts):
    print(f"  {categories[i]}: {count}")

#normalize to [0, 1]
X = X.astype('float32') / 255.0
X = X.reshape(-1, 28, 28, 1)

#split with shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

print(f"\nTraining: {len(X_train)}, Test: {len(X_test)}")

#bigger model with dropout
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)

#test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nOverall test accuracy: {test_acc * 100:.1f}%")

#per-class accuracy
print("\nPer-class test accuracy:")
for i, cat in enumerate(categories):
    mask = y_test == i
    if mask.sum() > 0:
        acc = model.evaluate(X_test[mask], y_test[mask], verbose=0)[1]
        print(f"  {cat}: {acc * 100:.1f}%")

#check predictions on test set
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
pred_counts = np.bincount(y_pred, minlength=5)

print("\nPrediction distribution on test set:")
for i, count in enumerate(pred_counts):
    print(f"  {categories[i]}: {count} ({count / len(y_pred) * 100:.1f}%)")

if pred_counts.max() > len(y_pred) * 0.4:
    print("\n⚠️ WARNING: Model is biased! Rerun training.")
else:
    print("\n✅ Model looks balanced!")

model.save("mythical_draw_model.keras")
print("\nModel saved!")