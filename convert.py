import ast
import os
import subprocess
import sys

import tensorflow as tf
from torchinfo import summary
from torchvision import models

sys.path.append("automl/efficientnetv2")
import effnetv2_model
import preprocessing

MODEL_SIZE = "l"  # @param  {"s", "m", "l"}
MODEL = f"efficientnetv2-{MODEL_SIZE}"

MODEL_PATH = "automl/efficientnetv2"


def download(m):
    ckpt_dir = os.path.join(MODEL_PATH, m)
    if m not in os.listdir(MODEL_PATH):
        subprocess.run(
            [
                "wget",
                f"https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/{m}.tgz",
                "-P",
                MODEL_PATH,
            ]
        )
        subprocess.run(["tar", "zxf", f"{ckpt_dir}.tgz", "-C", MODEL_PATH])
        subprocess.run(["rm", "-f", f"{ckpt_dir}.tgz"])
    ckpt_path = os.path.join(os.getcwd(), ckpt_dir)
    return ckpt_path


# Download checkpoint
def downloadCheckpoint():
    ckpt_path = download(MODEL + "-21k-ft1k")
    if tf.io.gfile.isdir(ckpt_path):
        ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    return ckpt_path


# Download label map file
def downloadLabelMapFile(labels_map):
    if labels_map not in os.listdir():
        subprocess.run(
            [
                "wget",
                "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt",
                "-O",
                labels_map,
            ]
        )


# Download images
def downloadImage(image_file):
    if image_file not in os.listdir():
        subprocess.run(
            [
                "wget",
                "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG",
                "-O",
                image_file,
            ]
        )


# Build model
def buildModel(ckpt_path):
    tf.keras.backend.clear_session()
    tf_model = effnetv2_model.EffNetV2Model(model_name=MODEL)
    _ = tf_model(tf.ones([1, 224, 224, 3]), training=False)
    tf_model.load_weights(ckpt_path)
    cfg = tf_model.cfg
    return tf_model, cfg


# Run inference for a given image
def runInference(image_file, cfg, tf_model):
    image = tf.io.read_file(image_file)
    image = preprocessing.preprocess_image(
        image, cfg.eval.isize, is_training=False, augname=cfg.data.augname
    )
    logits = tf_model(tf.expand_dims(image, 0), False)
    return logits


# Output classes and probability
def outputClassesAndProbability(logits, labels_map):
    pred = tf.keras.layers.Softmax()(logits)
    idx = tf.argsort(logits[0])[::-1][:5].numpy()

    classes = ast.literal_eval(open(labels_map, "r").read())
    for i, id in enumerate(idx):
        print(f"top {i+1} ({pred[0][id]*100:.1f}%):  {classes[id]} ")


def getTFModelStructure(tf_model: tf.keras.Model):
    structure = ""
    layers = [tf_model.layers[-3]] + tf_model.layers[:-3] + tf_model.layers[-2:]
    for layer in layers:
        for var in layer.weights:
            structure += var.name[17:] + " " + str(var.numpy().shape) + "\n"
        structure += "\n"
    return structure


def getTorchModelStructure(torch_model: models.EfficientNet):
    structure = ""
    for name, tensor in torch_model.state_dict().items():
        shape = tensor.to("cpu").numpy().shape
        if shape:
            structure += name + " " + str(shape) + "\n"
    return structure


def tf_main():
    ckpt_path = downloadCheckpoint()

    labels_map = "labels_map.txt"
    downloadLabelMapFile(labels_map)

    image_file = "panda.jpg"
    downloadImage(image_file)

    tf_model, cfg = buildModel(ckpt_path)

    logits = runInference(image_file, cfg, tf_model)

    outputClassesAndProbability(logits, labels_map)

    tf_model.summary(expand_nested=True, show_trainable=True)

    with open(f"tf_model-{MODEL_SIZE}_structure.txt", mode="w") as f:
        f.write(getTFModelStructure(tf_model))


def torch_main():
    torch_model: models.EfficientNet
    if MODEL_SIZE == "s":
        torch_model = models.efficientnet_v2_s()
    elif MODEL_SIZE == "m":
        torch_model = models.efficientnet_v2_m()
    elif MODEL_SIZE == "l":
        torch_model = models.efficientnet_v2_l()
    else:
        raise ValueError(
            f"Parameter MODEL_SIZE expect {{'s','m','l'}}, got {MODEL_SIZE}."
        )
    summary(
        torch_model,
        input_size=(1, 3, 224, 224),
        col_names=["kernel_size", "num_params"],
    )

    with open(f"torch_model-{MODEL_SIZE}_structure.txt", mode="w") as f:
        f.write(getTorchModelStructure(torch_model))


def convertWeightsFromTFToTorch():
    # Build TensorFlow model
    ckpt_path = downloadCheckpoint()
    tf_model, cfg = buildModel(ckpt_path)

    # # Get weights as NumPy array
    def weightsGenerator(tf_model: effnetv2_model.EffNetV2Model):
        layers = [tf_model.layers[-3]] + tf_model.layers[:-3] + tf_model.layers[-2:]
        for layer in layers:
            for var in layer.weights:
                yield (var.name, var.numpy())

    # Build PyTorch model
    torch_model: models.EfficientNet
    if MODEL_SIZE == "s":
        torch_model = models.efficientnet_v2_s()
    elif MODEL_SIZE == "m":
        torch_model = models.efficientnet_v2_m()
    elif MODEL_SIZE == "l":
        torch_model = models.efficientnet_v2_l()
    else:
        raise ValueError(
            f"Parameter MODEL_SIZE expect {{'s','m','l'}}, got {MODEL_SIZE}."
        )

    # Overwrite weights
    for (torch_layer_name, torch_weights), (tf_layer_name, tf_weights) in zip(
        torch_model.state_dict().items(), weightsGenerator(tf_model)
    ):
        # | TensorFlow | PyTorch |
        # ------------------------
        # | kernel | weight |  ->  conv
        # |  bias  |  bias  |  ->  conv
        # | gamma  | weight |  ->  batch_normalization
        # |  beta  |  bias  |  ->  batch_normalization
        # | moving_mean | running_mean |
        # | moving_variance | running_variance |
        assert (
            (
                "conv" in tf_layer_name
                and "kernel" in tf_layer_name
                and "weights" in torch_layer_name
            )
            or (
                "conv" in tf_layer_name
                and "bias" in tf_layer_name
                and "bias" in torch_layer_name
            )
            or (
                "normalization" in tf_layer_name
                and "gamma" in tf_layer_name
                and "weight" in torch_layer_name
            )
            or (
                "normalization" in tf_layer_name
                and "beta" in tf_layer_name
                and "bias" in torch_layer_name
            )
            or (
                "normalization" in tf_layer_name
                and "moving_mean" in tf_layer_name
                and "running_mean" in torch_layer_name
            )
            or (
                "normalization" in tf_layer_name
                and "moving_variance" in tf_layer_name
                and "running_variance" in torch_layer_name
            )
        )


if __name__ == "__main__":
    # tf_main()
    torch_main()
    # convertWeightsFromTFToTorch()