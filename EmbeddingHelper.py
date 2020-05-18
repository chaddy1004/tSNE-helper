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


class EmbeddingHelper():
    def __init__(self, encoder, data_generator, embeddings_dir):
        self.encoder = encoder
        self.data_generator = data_generator
        self.embeddings_dir = embeddings_dir
        self.metadata_filename = None
        self.sprite_filename = None
        self.embeddings = None
        self.data_array = None
        self.labels = None
        self.sprite_shape = None
        self._generate_embeddings()  # initialize the embeddings, data, and labels
        self._create_metadata()
        return

    def _generate_embeddings(self) -> None:
        embeddings_list = []
        data_array = None
        label_list = []
        counter = 0
        for train_data, label in self.data_generator:
            if data_array is None:
                data_array = train_data.numpy()
            else:
                data_array = np.append(data_array, train_data.numpy(), axis=0)
            embeddings = self.encoder.predict(train_data)
            for i, embedding in enumerate(embeddings):
                embeddings_list.append(embedding)
                label_list.append(label[i])
                counter += 1
        self.embeddings = np.array(embeddings_list)
        self.data_array = data_array
        self.labels = label_list

    def _create_metadata(self, filename="metadata.tsv") -> None:
        self.metadata_filename = filename
        metadata_filename = os.path.join(self.embeddings_dir, self.metadata_filename)
        with open(metadata_filename, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(self.labels):
                f.write(f"{index}\t{label}\n")
        return

    def _preprocess_data(self) -> Union[np.ndarray, Sequence[np.ndarray]]:
        """
        Function for preprocessing image data.
        For now, it normalizes the image, and reverses black and white
        Feel free to change this function for your own use
        :return: Processed image
        """
        # normalizing the image into 0-1
        copy = self.data_array / np.max(self.data_array)
        # reversing black and white to have white background and black content
        copy = 1 - copy
        return copy

    def create_sprite(self, filename="sprites.png") -> None:
        self.sprite_filename = filename
        n_data = self.data_array.shape[0]
        height = self.data_array.shape[1]
        width = self.data_array.shape[2]
        channel = self.data_array.shape[-1]
        n_plots = int(np.ceil(np.sqrt(n_data)))
        data_list_processed = self._preprocess_data()
        sprite_canvas = np.ones(((height * n_plots), (width * n_plots), channel))
        for row in range(n_plots):
            for col in range(n_plots):
                data_idx = row * n_plots + col
                if data_idx < n_data:
                    img = data_list_processed[data_idx, ...]
                    sprite_canvas[row * height:(row + 1) * height, col * width:(col + 1) * width, ...] = img
        alpha_channel = np.zeros_like(sprite_canvas)[..., [0]]
        alpha_channel[np.nonzero(sprite_canvas < 1)] = 1
        sprites_filename = os.path.join(self.embeddings_dir, self.sprite_filename)
        if channel == 0 or channel == 1:  # if it is grayscale, stack the images on top of each other
            sprite_png = np.concatenate([sprite_canvas, sprite_canvas, sprite_canvas, alpha_channel], axis=-1)
            sprite_png = (sprite_png * 255).astype(np.uint8)
        elif channel == 3:  # if it is RGB
            sprite_png = np.concatenate([sprite_canvas, alpha_channel], axis=-1)
        else:
            raise ValueError("Invalid image channel size. It must be none, 1 or 3")
        Image.fromarray(sprite_png).save(sprites_filename, "PNG")
        self.sprite_shape = (height, width, 4)
        return

    def to_tensorboard(self, checkpoint_filename="my-model.ckpt", embedding_name="z_tf") -> None:
        tf1.reset_default_graph()
        z = tf.Variable(self.embeddings, name=embedding_name)
        saver = tf.compat.v1.train.Saver(var_list=[z])
        ckpt_filename = os.path.join(self.embeddings_dir, checkpoint_filename)

        projector_config = ProjectorConfig()
        projector_config.model_checkpoint_path = ckpt_filename
        embeddings = projector_config.embeddings.add()

        embeddings.tensor_name = embedding_name

        # metadata_filename = os.path.join(self.embeddings_dir, self.metadata_filename)
        embeddings.metadata_path = self.metadata_filename

        # only add sprite if the the sprite is created in the first place
        if self.sprite_filename is not None:
            embeddings.sprite.image_path = os.path.join(self.embeddings_dir, self.sprite_filename)
            embeddings.sprite.image_path = self.sprite_filename
            embeddings.sprite.single_image_dim.extend(self.sprite_shape)

        visualize_embeddings(logdir=self.embeddings_dir, config=projector_config)
        saver.save(sess=None, global_step=None, save_path=ckpt_filename)

    def tsne_plot(self, labels, colors, filename="tsne.png", show=False) -> None:
        if len(labels) != len(colors):
            raise ValueError("The list of labels and colours should be the same!")
        filename_abs = os.path.join(self.embeddings_dir, filename)
        X_embedded = tsne(n_components=2).fit_transform(self.embeddings)
        plt.figure(figsize=(6, 5))

        for i, label in enumerate(self.labels):
            label = label.numpy()
            plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c=colors[label], label=int(label))
        lines = []

        for i, color in enumerate(colors):
            line = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=color)
            lines.append(line)

        plt.legend(lines, labels, numpoints=1, loc=1)
        plt.savefig(fname=filename_abs, format='png')
        if show:
            plt.show()
