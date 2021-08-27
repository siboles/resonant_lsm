import unittest
import tempfile
import shutil
import os

from resonant_lsm import generate_images


class SegmenterTests(unittest.TestCase):
    @classmethod
    def set_up_class(cls):
        cls._imageRootDir, cls._regions = generateImages.generateTestImages(number=2,
                                                                            background_noise=0.2,
                                                                            speckle_noise=0.1,
                                                                            spacing=0.1,
                                                                            output=tempfile.mkdtemp())

    @classmethod
    def tear_down_class(cls):
        shutil.rmtree(cls._imageRootDir)

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
