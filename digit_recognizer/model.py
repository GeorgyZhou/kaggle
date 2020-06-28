import tensorflow as tf
import pandas as pd

TRAIN_DATA_FILE_PATH = "data/train.csv"
TEST_DATA_FILE_PATH = "data/test.csv"

def _load_data():
    train_df = pd.read_csv(TRAIN_DATA_FILE_PATH)
    test_df = pd.read_csv(TEST_DATA_FILE_PATH)
    print(train_df.head())


def _get_train_strategy():
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

def train():
    strategy = _get_train_strategy()
    with strategy.scope():
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)])

        predictions = model(x_train[:1]).numpy()

        tf.nn.softmax(predictions).numpy()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)
        loss_fn(y_train[:1], predictions).numpy()

        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)

        model.evaluate(x_test, y_test, verbose=2)

if __name__ == "__main__":
    # train()
    _load_data()