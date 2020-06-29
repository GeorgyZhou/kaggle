import tensorflow as tf
import pandas as pd

TRAIN_DATA_FILE_PATH = "/data/kaggle/digit_recognizer/train.csv"
TEST_DATA_FILE_PATH = "/data/kaggle/digit_recognizer/test.csv"


def prepare_datasets():
    train_df = pd.read_csv(TRAIN_DATA_FILE_PATH)
    test_df = pd.read_csv(TEST_DATA_FILE_PATH)
    label = train_df.pop("label")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df.values / 255.0, label.values))
    train_rows = len(test_df)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_df.values / 255.0)
    return train_dataset, test_dataset, train_rows


def get_train_strategy():
    print("Tensorflow version: " + tf.__version__)
    # Detect hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy for hardware
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.master())
    elif len(gpus) > 0:
        print(type(gpus[0]))
        strategy = tf.distribute.MirroredStrategy(gpus)  # this works for 1 to multiple GPUs
        # strategy = tf.distribute.MirroredStrategy([":sdfasdfasdf"])  # this works for 1 to multiple GPUs
        print('Running on ', len(gpus), ' GPU(s) ')
    else:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on CPU')
    # How many accelerators do we have ?
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy


def train(train_dataset, row_count):
    train_dataset = train_dataset.shuffle(row_count).batch(1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(784, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     reduction=tf.keras.losses.Reduction.SUM),
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=1)
    return model


def predict(model, test_dataset):
    predictions = model.predict(test_dataset)
    print("===== Predictions =====")
    print(predictions)


if __name__ == "__main__":
    train_ds, test_ds, train_rows = prepare_datasets()
    print(test_ds.element_spec)
    train_strategy = get_train_strategy()
    with train_strategy.scope():
        trained_model = train(train_ds, train_rows)
        # predict(trained_model, test_ds)
