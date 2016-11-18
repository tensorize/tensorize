import unittest

import test_utils


def create_model(model_name, height, width, channels, classes):
    filename = "%s_%dx%dx%d_%d" % (model_name, height, width, channels, classes)
    meta_path = filename + ".meta"
    proto_path = filename + ".pb"
    return meta_path, proto_path


class TestModelFactories(unittest.TestCase):

    # def test_vgg16(self):
    #     test_utils.test_mnist(*create_model('vgg', 28, 28, 1, 10), epochs=6, batch_size=256)

    def test_lenet(self):
        test_utils.test_mnist(*create_model('lenet', 28, 28, 1, 10), epochs=10, batch_size=32)

    # def test_lenet_optimize(self):
    #     test_utils.test_mnist(*create_model('lenet_optimized', 28, 28, 1, 10), epochs=3, batch_size=1024)

    # def test_mnist(self):
    #     test_utils.test_mnist(*create_model('simple', 28, 28, 1, 10))
