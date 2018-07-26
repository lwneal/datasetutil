"""
 A Dataset file has the extension .dataset
 If it begins with the two byte sequence 0x1F8B then it is compressed
 Otherwise it is plain-text
 To decompress it, cat filename.dataset | gzip

 A decompressed .dataset file is a file in the NDJSON format.
 That is, it is a newline-delimited text file.
 Each line is a JSON key-value dictionary representing one example.

 Eg.
  {"filename": "foo/bar.jpg", "baz": 1}
  {"filename": "foo/boo.jpg", "baz": -1}
  {"filename": "baz/foo.jpg", "baz": 2, "color": "blue"}

 A key that exists in any example must exist in all examples (with the exception of "fold").
 Any property ending with "filename" should be a valid relative path string.
 Paths should be relative to the parent directory of the .dataset file at the time it is generated.
 Any property starting with the prefix "is_" or "has_" should be of Boolean type.
 The "fold" property, if included, must be a string. If not included, it defaults to "train"
 All other properties can be interpreted freely by Converter objects.

 The purpose of this format is to be simple to load, and easy to convert into other formats
"""
import random
import os
import json
import numpy as np
import copy

DEFAULT_FOLD = 'train'


class DatasetFile(object):
    def __init__(self, input_filename, example_count=None):
        input_filename = os.path.expanduser(input_filename)
        self.data_dir = os.path.dirname(input_filename)
        self.name = os.path.split(input_filename)[-1].replace('.dataset', '')

        data = open(input_filename).read()
        if data.startswith(chr(0x1F) + chr(0x8B)):
            print("Decompressing gzip file size {}".format(len(data)))
            data = data.decode('zlib')
        lines = data.strip().splitlines()
        self.examples = [json.loads(l) for l in lines]
        self.folds = get_folds(self.examples)
        if example_count:
            print("Randomly selecting {} examples".format(example_count))
            for fold in self.folds:
                random.shuffle(self.folds[fold])
                self.folds[fold] = self.folds[fold][:example_count]

        print("Dataset {} contains {} examples:".format(self.name, self.count()))
        for name in self.folds:
            print("\tFold '{}': {} examples".format(name, self.count(name)))

    def __add__(self, other):
        summed = copy.copy(self)
        for other_example in other.examples:
            summed.examples.append(other_example)
        summed.name = '-'.join([self.name, other.name])
        print("Combined dataset {} contains {} examples".format(
            summed.name, len(summed.examples)))
        return summed

    def _random_idx(self, fold):
        example_count = len(self.folds[fold])
        return np.random.randint(0, example_count)

    def count(self, fold=None):
        if fold:
            return len(self.folds[fold])
        return len(self.examples)

    def get_example(self, fold='train', idx=None, required_class=None):
        while True:
            if idx is None:
                idx = self._random_idx(fold)
            if required_class:
                # TODO: replace this rejection sampling scheme with something efficient
                if self.folds[fold][idx]['label'] != required_class:
                    continue
            return self.folds[fold][idx]

    def get_all_examples(self, fold='train'):
        for example in self.folds[fold]:
            yield example

    def get_batch(self, fold='train', batch_size=16, required_class=None):
        examples = []
        for i in range(batch_size):
            examples.append(self.get_example(fold, required_class=required_class))
        return examples

    def get_all_batches(self, fold='train', batch_size=16, shuffle=True, last_batch=False):
        examples = self.folds[fold]
        indices = list(range(len(examples)))
        if shuffle:
            random.shuffle(indices)
        batch = []
        for idx in indices:
            batch.append(self.get_example(fold, idx=idx))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if last_batch and len(batch) > 0:
            yield batch


def get_folds(examples):
    folds = {}
    for e in examples:
        fold = e.get('fold', DEFAULT_FOLD)
        if fold not in folds:
            folds[fold] = []
        folds[fold].append(e)
    return folds
