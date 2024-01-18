import tensorflow as tf


def small() -> tf.keras.Model:
    m = tf.keras.models.Sequential()

    m.add(tf.keras.layers.Conv2D(
        input_shape=(28, 28, 1),
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_last'
    ))
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='MaxPool1'
    ))

    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(10, name='FC1'))

    return m


def medium() -> tf.keras.Model:
# def small() -> tf.keras.Model:
    m = tf.keras.models.Sequential()

    m.add(tf.keras.layers.Conv2D(
        input_shape=(28, 28, 1),
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv1'
    ))
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='MaxPool1'
    ))

    m.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_last'
    ))
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='MaxPool2'
    ))

    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(10, name='FC1'))

    return m


def large() -> tf.keras.Model:
# def medium() -> tf.keras.Model:
    m = tf.keras.models.Sequential()

    m.add(tf.keras.layers.Conv2D(
        input_shape=(28, 28, 1),
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv1'
    ))
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='MaxPool1'
    ))

    m.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv2'
    ))
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='MaxPool2'
    ))

    m.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_last'
    ))

    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(256, activation='relu', name='FC1'))
    m.add(tf.keras.layers.Dense(10, name='FC2'))

    return m


# def large() -> tf.keras.Model:
#     m = tf.keras.models.Sequential()

#     m.add(tf.keras.layers.Conv2D(
#         input_shape=(28, 28, 1),
#         filters=32,
#         kernel_size=3,
#         strides=(1, 1),
#         padding='same',
#         activation='relu',
#         name='Conv1'
#     ))
#     m.add(tf.keras.layers.MaxPool2D(
#         pool_size=(2, 2),
#         name='MaxPool1'
#     ))

#     m.add(tf.keras.layers.Conv2D(
#         filters=32,
#         kernel_size=3,
#         strides=(1, 1),
#         padding='same',
#         activation='relu',
#         name='Conv2'
#     ))
#     m.add(tf.keras.layers.MaxPool2D(
#         pool_size=(2, 2),
#         name='MaxPool2'
#     ))

#     m.add(tf.keras.layers.Conv2D(
#         filters=32,
#         kernel_size=3,
#         strides=(1, 1),
#         padding='same',
#         activation='relu',
#         name='Conv_last'
#     ))

#     m.add(tf.keras.layers.Flatten())
#     m.add(tf.keras.layers.Dense(512, activation='relu', name='FC1'))
#     m.add(tf.keras.layers.Dense(256, activation='relu', name='FC2'))
#     m.add(tf.keras.layers.Dense(10, name='FC3'))

#     return m
