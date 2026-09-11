from tensorflow.keras import layers, models

from .labels import CLASS_NAMES
from .preprocessing import DEFAULT_IMAGE_SIZE, DEFAULT_NUM_FRAMES


def build_cnn_bilstm(
    num_frames: int = DEFAULT_NUM_FRAMES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_classes: int = len(CLASS_NAMES),
) -> models.Model:
    input_shape = (num_frames, image_size[1], image_size[0], 3)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.TimeDistributed(
                layers.Conv2D(32, kernel_size=3, activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.MaxPooling2D(pool_size=2)),
            layers.TimeDistributed(
                layers.Conv2D(64, kernel_size=3, activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.MaxPooling2D(pool_size=2)),
            layers.TimeDistributed(
                layers.Conv2D(128, kernel_size=3, activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
