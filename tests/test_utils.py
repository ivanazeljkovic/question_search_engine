import shutil
import unittest
from utils import *


class TestUtils(unittest.TestCase):

    def test_check_does_file_exist(self):
        # file exists
        file_path = 'file.txt'
        with open(file_path, 'w') as file:
            file.write('text')
        self.assertIsInstance(check_does_file_exist(file_path), bool)
        self.assertEqual(check_does_file_exist(file_path), True)

        # file doesn't exist
        os.remove(file_path)
        self.assertIsInstance(check_does_file_exist(file_path), bool)
        self.assertEqual(check_does_file_exist(file_path), False)

    def test_check_does_dir_exist(self):
        # directory exists
        dir_path = 'dir'
        os.mkdir(dir_path)
        self.assertIsInstance(check_does_dir_exist(path=dir_path, create_dir=False), bool)
        self.assertEqual(check_does_dir_exist(path=dir_path, create_dir=False), True)

        # dir doesn't exist, without creating new one
        shutil.rmtree(dir_path)
        self.assertIsInstance(check_does_dir_exist(path=dir_path, create_dir=False), bool)
        self.assertEqual(check_does_dir_exist(path=dir_path, create_dir=False), False)

        # dir doesn't exist, with creating new one
        self.assertIsInstance(check_does_dir_exist(path=dir_path, create_dir=True), bool)
        self.assertEqual(check_does_dir_exist(path=dir_path, create_dir=True), True)
        shutil.rmtree(dir_path)

    def test_serialize_data(self):
        data = [1, 2, 3]

        # serialization with file path
        path = 'list.pkl'
        self.assertEqual(serialize_data(data, path), None)
        self.assertEqual(check_does_file_exist(path), True)
        self.assertEqual(deserialize_data(path), data)
        os.remove(path)

        # serialization with file in non existent directory
        path = 'dir/list.pkl'
        self.assertEqual(serialize_data(data, path), None)
        self.assertEqual(check_does_dir_exist(path='dir', create_dir=False), True)
        self.assertEqual(check_does_file_exist(path), True)
        self.assertEqual(deserialize_data(path), data)
        shutil.rmtree('dir')

    def test_deserialize_data(self):
        data = [1, 2, 3]
        path = 'list.pkl'
        serialize_data(data, path)

        # deserialization from existing path
        self.assertIsInstance(deserialize_data(path), List)
        self.assertEqual(deserialize_data(path), data)
        os.remove(path)

        # deserialization from non existent path
        with self.assertRaises(FileNotFoundError):
            deserialize_data(path)


if __name__ == '__main__':
    unittest.main()
