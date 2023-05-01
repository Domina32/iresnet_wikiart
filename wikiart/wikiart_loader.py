import ast
import os
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as fn
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm


def resize_img(img, new_width, new_height):
    """Resize an image using new  width and height"""
    new_points = (new_width, new_height)

    return fn.resize(img, new_points, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)


class WikiartDataset(IterableDataset):
    batch_size = 1
    base_path = "."
    in_shape = (300, 300, 3)
    chosen_label = "genre"
    files = []
    number_of_files = 0
    file_index_map: Dict[str, list[int]] = {}
    file_length_map: Dict[str, int] = {}
    available_files = []

    def __init__(
        self, batch_size=1, chosen_label: Union[Literal["artist"], Literal["genre"], Literal["style"]] = "genre"
    ):
        self.files = os.listdir(f"{self.base_path}/wikiart/data/csv/")
        self.number_of_files = len(self.files)
        self.available_files = self.files

        for file_name in self.files:
            self.file_index_map[file_name] = []

        self.generator = self.__read_tf_dataset()
        self.batch_size = batch_size
        self.chosen_label = chosen_label

    def __iter__(self):
        return self.generator

    # Use generator directly
    def __read_data(self):
        step = 0
        batch = 0

        with tqdm(total=self.batch_size, desc="Loading data for batch 0") as pbar:
            for row in WrappedGenerator(self.__get_random_row_from_set):
                try:
                    images = (
                        row["image"]
                        .map(
                            lambda img: torchvision.io.decode_image(
                                torch.tensor(np.frombuffer(ast.literal_eval(img).get("bytes"), dtype=np.uint8))
                            )
                        )
                        .values
                    )
                    labels = row[self.chosen_label].values
                except Exception as e:
                    print(e)
                    pbar.update(1)
                    step += 1
                    continue

                if step < self.batch_size:
                    pbar.update(1)
                    step += 1
                else:
                    pbar.reset()
                    batch += 1
                    step = 0
                    pbar.set_description(f"Loading data for batch {batch}")
                yield (
                    np.asarray([resize_img(image, self.in_shape[0], self.in_shape[1]) for image in images]),
                    np.asarray(labels),
                )

    def __read_tf_dataset(self):
        step = 0
        batch = 0

        with tqdm(total=self.batch_size, desc="Loading data for batch 0") as pbar:
            for row in WrappedGenerator(self.__get_random_row_from_set):
                try:
                    images = (
                        row["image"]
                        .map(
                            lambda img: resize_img(
                                torchvision.io.decode_image(
                                    torch.tensor(np.frombuffer(ast.literal_eval(img).get("bytes"), dtype=np.uint8))
                                ),
                                self.in_shape[0],
                                self.in_shape[1],
                            )
                        )
                        .values
                    )
                    labels = row[self.chosen_label].values

                except Exception as e:
                    print(e)
                    pbar.update(1)
                    step += 1
                    continue

                if step < self.batch_size:
                    pbar.update(1)
                    step += 1
                else:
                    pbar.reset()
                    batch += 1
                    step = 0
                    pbar.set_description(f"Loading data for batch {batch}")

                yield images[0], labels[0]

    def __get_random_row_from_file(self, file_name: str, pbar: Optional[tqdm]):
        df: pd.DataFrame
        random_index: int
        if file_name not in self.file_length_map:
            df = pd.read_csv(f"{self.base_path}/wikiart/data/csv/" + file_name)
            self.file_length_map[file_name] = df.shape[0]
            random_index = np.random.randint(1, self.file_length_map[file_name])
            df = df[random_index : random_index + 1]
            self.file_index_map[file_name].append(0)
            pbar.update(1)
        else:
            choices = [
                index
                for index in range(1, self.file_length_map[file_name])
                if index not in self.file_index_map[file_name]
            ]
            random_index = np.random.choice(choices)
            df = pd.read_csv(f"{self.base_path}/wikiart/data/csv/" + file_name, skiprows=(1, random_index), nrows=1)

        self.file_index_map[file_name].append(random_index)

        return df

    def __get_random_row_from_set(self):
        with tqdm(total=len(self.files), desc="Accessed files/Total files") as pbar:
            while len(self.available_files) > 0:
                self.available_files = [
                    file_name
                    for file_name in self.files
                    if file_name not in self.file_length_map
                    or len(self.file_index_map[file_name]) < self.file_length_map[file_name]
                ]
                random_file = np.random.choice(self.available_files)

                yield self.__get_random_row_from_file(random_file, pbar)


class WrappedGenerator:
    def __init__(self, generator: callable, *args, **kwargs):
        self._generator = generator(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._generator)
        except Exception as e:
            print(e)
            return next(self._generator)
