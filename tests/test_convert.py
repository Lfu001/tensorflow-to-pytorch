import os
import sys
import unittest

import numpy as np
import tensorflow as tf
import torch
import torchvision
from convert import buildModel, downloadCheckpoint, outputClassesAndProbability

sys.path.append("automl/efficientnetv2")
import preprocessing


class TestConvert(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.MODEL_SIZE = "s"  # @param  {"s", "m", "l"}
        self.MODEL_PRETRAIN = "21k-ft1k"  # @param {"21k", "21k-ft1k"}
        self.MODEL = f"efficientnetv2-{self.MODEL_SIZE}"

    def setUp(self) -> None:
        super().setUp()

        ckpt = downloadCheckpoint(self.MODEL + "-" + self.MODEL_PRETRAIN)
        self.tf_model, cfg = buildModel(self.MODEL, ckpt)
        self.tf_img = tf.io.read_file("panda.jpg")
        self.tf_img = preprocessing.preprocess_image(
            self.tf_img, cfg.eval.isize, is_training=False, augname=cfg.data.augname
        )
        print("model_name:", cfg.model.model_name)
        print("tf.expand_dims(self.tf_img, 0)", tf.expand_dims(self.tf_img, 0).shape)

        self.torch_model: torchvision.models.EfficientNet = torch.load(
            os.path.join("torch_models", f"{self.MODEL}-{self.MODEL_PRETRAIN}.pth")
        )
        self.torch_model.eval()
        self.pt_img = torchvision.transforms.ToTensor()(self.tf_img.numpy())
        self.pt_img_batch = self.pt_img[None]
        print("self.pt_img_batch.shape", self.pt_img_batch.shape)

        self.tf_logits: tf.Tensor = self.tf_model(
            tf.expand_dims(self.tf_img, 0), training=False
        )
        self.pt_logits: torch.Tensor = self.torch_model(self.pt_img_batch)

        self.diff = np.abs(self.tf_logits.numpy() - self.pt_logits.detach().numpy())
        self.tf_proba = tf.keras.layers.Softmax()(self.tf_logits.numpy())
        self.pt_proba = tf.keras.layers.Softmax()(self.pt_logits.detach().numpy())
        self.proba_diff = np.abs(self.tf_proba - self.pt_proba)

        # tf_weights = ""
        # for layer in self.tf_model.layers:
        #     for var in layer.weights:
        #         tf_weights += var.name[17:] + "\n" + str(var.numpy()) + "\n\n"
        # with open(f"tf_model-{self.MODEL_SIZE}_weights.txt", mode="w") as f:
        #     f.write(tf_weights)
        #     del tf_weights
        # torch_weights = ""
        # for name, tensor in self.torch_model.state_dict().items():
        #     shape = tensor.to("cpu").numpy().shape
        #     if shape:
        #         if len(shape) == 4:
        #             weight_string = str(tensor.detach().numpy().transpose(2, 3, 1, 0))
        #         elif len(shape) == 1:
        #             weight_string = str(tensor.detach().numpy())
        #         elif len(shape) == 2:
        #             weight_string = str(tensor.detach().numpy().transpose(1, 0))
        #         else:
        #             ValueError()
        #         torch_weights += name + "\n" + weight_string + "\n\n"
        # with open(f"torch_model-{self.MODEL_SIZE}_weights.txt", mode="w") as f:
        #     f.write(torch_weights)
        #     del torch_weights

    def tearDown(self) -> None:
        super().tearDown()
        print("=" * 20 + "\nTensorFlow model\n" + "=" * 20)
        outputClassesAndProbability(self.tf_logits.numpy(), "labels_map.txt")

        print("=" * 20 + "\nPyTorch model\n" + "=" * 20)
        outputClassesAndProbability(self.pt_logits.detach().numpy(), "labels_map.txt")
        print()

        print("diff:", np.mean(self.diff), np.std(self.diff))
        print("probability diff:", np.mean(self.proba_diff), np.std(self.proba_diff))

    def testImageEqual(self):
        tf_img = tf.expand_dims(self.tf_img, 0).numpy()
        torch_img = self.pt_img_batch.detach().numpy()
        self.assertEqual(tf_img.dtype, torch_img.dtype)
        self.assertEqual(tf_img.tolist(), torch_img.transpose(0, 2, 3, 1).tolist())

    def testModelWeightsDtype(self):
        torch_layers = (
            item
            for item in self.torch_model.state_dict().items()
            if item[1].numpy().shape
        )
        weights_type = set()
        for _, weight in torch_layers:
            weights_type.add(weight.numpy().dtype)
        print(weights_type)

    def testLogitsSize(self):
        tf_logit_shape = self.tf_logits.numpy().shape
        torch_logit_shape = self.pt_logits.detach().numpy().shape
        print(tf_logit_shape, torch_logit_shape)
        self.assertEqual(tf_logit_shape, torch_logit_shape)

    # def testLogitsDiff(self):
    #     for i, (tf_logit, torch_logit) in enumerate(
    #         zip(self.tf_logits.numpy()[0], self.pt_logits.detach().numpy()[0])
    #     ):
    #         self.assertAlmostEqual(tf_logit, torch_logit, msg=f"Index {i} failed.")

    def testProbaDiff(self):
        for i, (tf_proba, torch_proba) in enumerate(zip(self.tf_proba, self.pt_proba)):
            self.assertAlmostEqual(
                tf_proba.numpy()[0],
                torch_proba.numpy()[0],
                msg=f"Index {i} failed.",
                places=4,
            )


class TestModelS(TestConvert):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.MODEL_SIZE = "s"  # @param  {"s", "m", "l"}
        self.MODEL_PRETRAIN = "21k-ft1k"  # @param {"21k", "21k-ft1k"}
        self.MODEL = f"efficientnetv2-{self.MODEL_SIZE}"


class TestModelM(TestConvert):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.MODEL_SIZE = "m"  # @param  {"s", "m", "l"}
        self.MODEL_PRETRAIN = "21k-ft1k"  # @param {"21k", "21k-ft1k"}
        self.MODEL = f"efficientnetv2-{self.MODEL_SIZE}"


class TestModelL(TestConvert):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.MODEL_SIZE = "l"  # @param  {"s", "m", "l"}
        self.MODEL_PRETRAIN = "21k-ft1k"  # @param {"21k", "21k-ft1k"}
        self.MODEL = f"efficientnetv2-{self.MODEL_SIZE}"


if __name__ == "__main__":
    unittest.main()
