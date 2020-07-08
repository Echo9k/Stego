import tensorflow as tf
from typing import Tuple, Optional, Generator, Dict, List
from os import walk
from .Stego import Stego


class GetData(Stego):
    def __init__(self, dir_url: Dict[str, str] = None, class_names: [List] = None, img_directory: str = './image_data'):
        super().__init__(dir_url, class_names, img_directory)
        self.dir_url = dir_url
        self.class_names = self._folder_names(class_names, img_directory)
        self.img_directory = img_directory

    @staticmethod
    def _folder_names(class_names=None, img_directory=None):
        if class_names is None:
            try:
                _, directories, _ = next(walk(img_directory))
                return (None, directories)[len(directories) > 0]  # return directory names
            except StopIteration:
                'The attribute img_directory should be relative./"folder" or complete /.../"folder".'
        else:
            return class_names

    def _prevent_duplicates(self) -> Dict:
        requested_folders = set(self.dir_url)
        folders_in_directory = self._folder_names(img_directory=self.img_directory)
        try:
            to_download = requested_folders.symmetric_difference(folders_in_directory)
            print(f'Downloading: {to_download}.')
            return {i: self.dir_url.get(i) for i in to_download}
        except TypeError:
            ""
            return self.dir_url

    def download_unzip(self, as_generator=False) -> None or Generator:
        """
        Downloads data from the dir_url of the form {category, url}.
        Stores each folder under img_directory/category/

        PARAMETERS:
        :param as_generator: [default=False] Changes the behavior of the downloader. 
         If set to false it will automatically download all the folders from  "self.dir_url"
         and store them in folders with the keys of that directory.

        :return: None, or a generator if asGenerator is set to True.
        """

        def _mk_params(dir_key, file_url):
            params = {'fname': dir_key, 'origin': file_url,
                      'cache_subdir': self.img_directory + dir_key,
                      'hash_algorithm': 'auto', 'extract': True,
                      'archive_format': 'auto', 'cache_dir': None}
            return params

        to_download = self._prevent_duplicates()
        try:
            if not as_generator:
                for key, url in to_download.items():
                    f_params = _mk_params(key, url)
                    tf.keras.utils.get_file(**f_params)
            else:
                return (tf.keras.utils.get_file(key, url) for key, url in self.dir_url.items() if key in to_download)
        except TypeError:
            print(f"All the requested folders already exist on '{self.img_directory}'")

    def img_batch(self, batch_size: Optional[int] = 32,
                  target_size: Optional[Tuple] = (256, 256),
                  subset: Optional[str] = 'training', *,
                  validation_split: Optional[int] = 0.3,
                  class_mode: Optional[int] = 'categorical',
                  preprocessing_function=None) -> tf.keras.preprocessing.image.DirectoryIterator:
        """
        :int batch_size: size of the batches of data (default: 32)
        :tuple target_size: size of the output images.
        :str subset: `"training"` or `"validation"`.
        :float validation_split = A percentage of data to use as validation set.
        :str class_mode: Type of classification for this flow of data
            - binary:if there are only two classes
            - categorical: categorical targets,
            - sparse: integer targets,
            - input: targets are images identical to input images (mainly used to work with autoencoders),
            - None: no targets get yielded (only input images are yielded).
        """

        img_gen_params = {'featurewise_center': False,
                          'samplewise_center': True,
                          'featurewise_std_normalization': False,
                          'samplewise_std_normalization': True,
                          'zca_whitening': False,
                          'fill_mode': 'reflect',
                          'horizontal_flip': True,
                          'vertical_flip': True,
                          'validation_split': validation_split,
                          'preprocessing_function': preprocessing_function
                          }
        img_gen = tf.keras.preprocessing.image.ImageDataGenerator(**img_gen_params)

        self.class_names = self._folder_names()

        img_dir_params = {'directory': self.img_directory,
                          'image_data_generator': img_gen,
                          'target_size': target_size,
                          'color_mode': 'rgb',
                          'classes': self.class_names,
                          'class_mode': class_mode,
                          'batch_size': batch_size,
                          'shuffle': True
                          }
        print(subset + ':')

        return tf.keras.preprocessing.image.DirectoryIterator(**img_dir_params, subset=subset)
