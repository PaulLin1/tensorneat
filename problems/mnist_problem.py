import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from tensorneat.problem.base import BaseProblem
from jax import remat

class MNISTClassificationProblem(BaseProblem):
    jitable = True
    def __init__(self, val_split=0.2):
        super().__init__()
        data = np.load("mnist_train.npz")
        images, labels = data["images"], data["labels"]

        labels_onehot = np.eye(10)[labels.astype(int)]

        X_train, X_val, Y_train, Y_val = train_test_split(
            images, labels_onehot, test_size=val_split, random_state=42
        )

        # Convert all data to jax arrays for vmap
        self.X_train = jnp.array(X_train)
        self.Y_train = jnp.array(Y_train)
        self.X_val = jnp.array(X_val)
        self.Y_val = jnp.array(Y_val)

    @property
    def input_shape(self):
        return (784,)

    @property
    def output_shape(self):
        return (10,)


    def evaluate(self, state, randkey, act_func, params, batch_size=128, max_eval_samples=None):
        # Wrap act_func with remat to save memory
        remat_act_func = remat(act_func)

        # Optionally restrict to a subset for speed
        if max_eval_samples is not None:
            max_eval_samples = min(max_eval_samples, self.X_val.shape[0])
            X_val = self.X_val[:max_eval_samples]
            Y_val = self.Y_val[:max_eval_samples]
        else:
            X_val = self.X_val
            Y_val = self.Y_val

        num_samples = X_val.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size  # ceiling division

        preds_list = []
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch = X_val[start:end]
            batch_preds = jax.vmap(remat_act_func, in_axes=(None, None, 0))(state, params, batch)
            preds_list.append(batch_preds)

        preds = jnp.concatenate(preds_list, axis=0)

        eps = 1e-8
        loss = -jnp.mean(jnp.sum(Y_val * jnp.log(preds + eps), axis=-1))

        predicted_classes = jnp.argmax(preds, axis=-1)
        true_classes = jnp.argmax(Y_val, axis=-1)
        accuracy = jnp.mean(predicted_classes == true_classes)

        return accuracy


    def show(self, state, randkey, act_func, params, max_display=20):
        preds = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, self.X_val)

        predicted_classes = jnp.argmax(preds, axis=-1)
        true_classes = jnp.argmax(self.Y_val, axis=-1)

        for i in range(min(max_display, self.X_val.shape[0])):
            print(f"Input {i}: true={int(true_classes[i])}, predicted={int(predicted_classes[i])}")

        accuracy = jnp.mean(predicted_classes == true_classes)
        print(f"Validation accuracy: {float(accuracy):.4f}")


