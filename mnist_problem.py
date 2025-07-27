import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorneat.problem.base import BaseProblem


class MNISTClassificationProblem(BaseProblem):
    jitable = True

    def __init__(self):
        super().__init__()
        # Load MNIST dataset from tensorflow_datasets
        ds = tfds.load('mnist', split='train', batch_size=-1, as_supervised=True)
        images, labels = tfds.as_numpy(ds)

        # Normalize to [0, 1] and flatten
        X = images.astype(jnp.float32) / 255.0
        X = X.reshape((X.shape[0], -1))  # Flatten 28x28 -> 784

        # One-hot encode labels
        Y = jax.nn.one_hot(jnp.array(labels), num_classes=10)

        # Train/val split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.X_train = jnp.array(X_train)
        self.Y_train = jnp.array(Y_train)
        self.X_val = jnp.array(X_val)
        self.Y_val = jnp.array(Y_val)

    @property
    def input_shape(self):
        return (784,)  # 28x28 images flattened

    @property
    def output_shape(self):
        return (10,)  # 10 classes

    def evaluate(self, state, randkey, act_func, params):
        preds = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, self.X_val)

        eps = 1e-8
        loss = -jnp.mean(jnp.sum(self.Y_val * jnp.log(preds + eps), axis=-1))

        predicted_classes = jnp.argmax(preds, axis=-1)
        true_classes = jnp.argmax(self.Y_val, axis=-1)
        accuracy = jnp.mean(predicted_classes == true_classes)

        return accuracy

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        preds = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, self.X_val)
        predicted_classes = jnp.argmax(preds, axis=-1)
        true_classes = jnp.argmax(self.Y_val, axis=-1)

        for i in range(min(20, self.X_val.shape[0])):
            print(f"Input {i}: true={true_classes[i]}, predicted={predicted_classes[i]}")

        accuracy = jnp.mean(predicted_classes == true_classes)
        print(f"Validation accuracy: {accuracy}")
