import unittest
import numpy as np
import torch
from modules.utils import filter_detections_for_dumping


class TestFilterDetectionsForDumping(unittest.TestCase):
    def setUp(self):
        class Args:
            GEN_CONF_THRESH = 0.5
            GEN_TOPK = 100
            GEN_NMS = 0.5

        self.args = Args()


if __name__ == "__main__":
    unittest.main()
