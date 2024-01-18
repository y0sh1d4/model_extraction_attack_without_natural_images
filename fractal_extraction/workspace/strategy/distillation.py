import tensorflow as tf


class Distillation(tf.keras.Sequential):
    def train_step(self, batch):
        x, t = batch
        with tf.GradientTape() as tape:
            y_pred = tf.keras.activations.softmax(self(x, training=True))
            loss = self.compiled_loss(t, y_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        self.compiled_metrics.update_state(
            tf.one_hot(tf.argmax(t, axis=1), depth=t.shape[-1]), y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
