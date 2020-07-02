import csv
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

TRAIN_DATA_FILE_PATH = "/data/kaggle/digit_recognizer/train.csv"
TEST_DATA_FILE_PATH = "/data/kaggle/digit_recognizer/test.csv"
OUTPUT_FILE_PATH = "/data/kaggle/digit_recognizer/predictions.csv"


def prepare_datasets():
    train_df = pd.read_csv(TRAIN_DATA_FILE_PATH)
    test_df = pd.read_csv(TEST_DATA_FILE_PATH)
    label = train_df.pop("label")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df.values / 255.0, label.values))
    train_rows = len(train_df)
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


def train(train_dataset, sample_count):
    train_dataset = train_dataset.shuffle(sample_count).batch(1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(784, 1), dtype='float64'),
        tf.keras.layers.Dense(128, activation='relu', dtype='float64'),
        tf.keras.layers.Dense(256, activation='relu', dtype='float64'),
        tf.keras.layers.Dropout(0.2, dtype='float64'),
        tf.keras.layers.Dense(10, activation="softmax", dtype='float64')])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     reduction=tf.keras.losses.Reduction.SUM),
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=50)
    return model


def predict_class(model, test_dataset):
    test_dataset = test_dataset.batch(1)
    predictions = np.argmax(model.predict(test_dataset), axis=-1)
    print("===== Predictions =====")
    print(predictions)
    return predictions


def output_results(output_path, predictions):
    print("Writing results to " + output_path)
    with open(output_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["ImageId", "Label"])
        writer.writeheader()
        for index, prediction in enumerate(predictions):
            writer.writerow({"ImageId": index + 1, "Label": prediction})


if __name__ == "__main__":
    begin_time = datetime.datetime.now()
    train_ds, test_ds, train_count = prepare_datasets()
    train_strategy = get_train_strategy()
    with train_strategy.scope():
        trained_model = train(train_ds, train_count)
        results = predict_class(trained_model, test_ds)
        output_results(OUTPUT_FILE_PATH, results)
    print("Total running time: " + str(datetime.datetime.now() - begin_time))