import numpy as np
import os
from typing import Union, Sequence
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorboard.plugins.projector import ProjectorConfig, visualize_embeddings
from PIL import Image
from sklearn.manifold import TSNE as tsne
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt


def create_tsne_plot(embedding_array, label_list, embeddings_dir, filename="tsne.png"):
    filename_abs = os.path.join(embeddings_dir, filename)
    X_embedded = tsne(n_components=2).fit_transform(embedding_array)
    plt.figure(figsize=(6, 5))
    target_ids = ['PEN', 'FLEX', 'SCAP', 'ABD', 'IR', 'ER', 'DIAG', 'ROW', 'SLR']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange']

    for i, label in enumerate(label_list):
        label = label.numpy()
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c=colors[label], label=int(label))
    lines = []

    for i, color in enumerate(colors):
        line = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=color)
        lines.append(line)

    plt.legend(lines, target_ids, numpoints=1, loc=1)
    plt.savefig(fname=filename_abs, format='png')
    plt.show()


def create_embeddings(data_generator, encoder, embeddings_dir):
    embeddings_list = []
    data_array = None
    label_list = []
    counter = 0
    for train_data, label in data_generator:
        if data_array is None:
            data_array = train_data.numpy()
        else:
            data_array = np.append(data_array, train_data.numpy(), axis=0)
        # in this case, the label is simply the index of the data. This can be adapted for different data later on
        embeddings = encoder.predict(train_data)  # [0]
        for i, embedding in enumerate(embeddings):
            embeddings_list.append(embedding)
            label_list.append(label[i])
            counter += 1
    return np.array(embeddings_list), data_array, label_list


def preprocess_data(data_array: Union[np.ndarray, Sequence[np.ndarray]]) -> Union[np.ndarray, Sequence[np.ndarray]]:
    data_array = 1 - data_array
    return data_array


def create_sprite(data_array: np.ndarray, embeddings_dir: str,
                  filename="sprites.png") -> None:
    n_data = data_array.shape[0]
    height = data_array.shape[1]
    width = data_array.shape[2]
    channel = data_array.shape[-1]
    n_plots = int(np.ceil(np.sqrt(n_data)))
    data_list_processed = preprocess_data(data_array=data_array)
    sprite_canvas = np.ones(((height * n_plots), (width * n_plots), channel))
    for row in range(n_plots):
        for col in range(n_plots):
            data_idx = row * n_plots + col
            if data_idx < n_data:
                img = data_list_processed[data_idx, ...]
                sprite_canvas[row * height:(row + 1) * height, col * width:(col + 1) * width, ...] = img
    alpha_channel = np.zeros_like(sprite_canvas)[..., [0]]
    alpha_channel[np.nonzero(sprite_canvas < 1)] = 1
    sprites_filename = os.path.join(embeddings_dir, filename)
    if channel == 0:
        sprite_png = np.concatenate([sprite_canvas, sprite_canvas, sprite_canvas, alpha_channel], axis=-1)
        sprite_png = (sprite_png * 255).astype(np.uint8)
    else:
        sprite_png = np.concatenate([sprite_canvas, alpha_channel], axis=-1)
    Image.fromarray(sprite_png).save(sprites_filename, "PNG")
    return


def create_metadata(label_list: Union[np.ndarray, Sequence[np.ndarray]], embeddings_dir: str,
                    filename="metadata.tsv") -> None:
    metadata_filename = os.path.join(embeddings_dir, filename)
    with open(metadata_filename, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(label_list):
            f.write(f"{index}\t{label}\n")
    return


def embedding_dump(embeddings: Union[np.ndarray, Sequence[np.ndarray]], embeddings_dir: str,
                   checkpoint_filename="my-model.ckpt", metadata_filename="metadata.tsv",
                   sprite_filename="sprites.png",
                   embedding_name="z_tf") -> None:
    tf1.reset_default_graph()

    # you initialize a tf variable with embeddings as its initial values
    # initializer = tf.constant_initializer(embeddings)
    # z = tf1.get_variable(embedding_name, shape=embeddings.shape, initializer=initializer)
    z = tf.Variable(embeddings, name=embedding_name)
    saver = tf.compat.v1.train.Saver(var_list=[z])
    ckpt_filename = os.path.join(embeddings_dir, checkpoint_filename)
    # ckpt_filename = checkpoint_filename

    projector_config = ProjectorConfig()
    projector_config.model_checkpoint_path = embeddings_dir
    embeddings = projector_config.embeddings.add()

    embeddings.tensor_name = embedding_name

    # metadata_filename = os.path.join(embeddings_dir, metadata_filename)
    embeddings.metadata_path = metadata_filename

    # embeddings.sprite.image_path = os.path.join(embeddings_dir, sprite_filename)
    # embeddings.sprite.image_path = sprite_filename
    # embeddings.sprite.single_image_dim.extend([28, 28,4])

    visualize_embeddings(logdir=embeddings_dir, config=projector_config)
    saver.save(sess=None, global_step=None, save_path=ckpt_filename)
