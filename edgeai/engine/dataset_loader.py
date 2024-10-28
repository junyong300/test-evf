""" Dataset Loader producing data from tfrecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow.keras as hvd

def load_tfdataset(dataset_name, split) -> tf.data.Dataset:
    """Return a dataset loading files with TFRecords."""

    try:
        dataset = tfds.load(dataset_name, split=split)
    except Exception as e:
        print(e)
        file_pattern = os.path.join(dataset_name, '{}*'.format(split))
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    return dataset

class DatasetLoader(ABC):
    """
        Read tfrecords files or tf.dataset

    """

    def __init__(self, parse_record, batch_size, prep_func=None, training=False):
        self._parse_record = parse_record
        self._batch_size = batch_size
        self._prep_func = prep_func
        self._training = training

    @abstractmethod
    def build(self):
        """
            Return tf.data.Dataset which is compatible to `.fit`.
        """


class TFHVDDatasetLoader(DatasetLoader):
    """
        
        TensorFlow-HVD Dataset Loader

    """

    def __init__(
            self,
            parse_record,
            batch_size,
            prep_func=None,
            training=False,
            buffer_size_for_shuffling=1024,
            use_cache=False):
        super(TFHVDDatasetLoader, self).__init__(parse_record, batch_size, prep_func, training)
        self._num_gpus = hvd.size()
        self._buffer_size_for_shuffling = buffer_size_for_shuffling
        self._use_cache = use_cache

    def build(self, dataset):
        options = tf.data.Options()
        options.experimental_optimization.map_parallelization = True
        dataset = dataset.with_options(options)

        # Handling multiple gpus
        if self._num_gpus > 1:
            dataset = dataset.shard(self._num_gpus, hvd.rank())

        if self._use_cache:
            dataset = dataset.cache()

        if self._training:
            dataset = dataset.shuffle(buffer_size=self._buffer_size_for_shuffling)
            dataset = dataset.repeat()

        dataset = dataset.map(self._parse_record,
                              num_parallel_calls=tf.data.AUTOTUNE)

        if self._num_gpus > 1:
            dataset = dataset.batch(self._batch_size,
                                    drop_remainder=self._training)
        else:
            dataset = dataset.batch(self._batch_size * self._num_gpus,
                                    drop_remainder=self._training)

        if self._prep_func is not None:
            dataset = dataset.map(self._prep_func, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
