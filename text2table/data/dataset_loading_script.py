# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Loading script for MIMIC3 dataset."""


import csv
import json
from multiprocessing.sharedctypes import Value
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
 @article{johnson_pollard_shen_lehman_feng_ghassemi_moody_szolovits_anthony celi_mark_et al._2016, 
 title={Mimic-III, a freely accessible Critical Care Database}, 
 url={https://www.nature.com/articles/sdata201635#citeas}, DOI={10.1038/sdata.2016.35},
 journal={Nature Scientific Data}, 
 author={Johnson, Alistair E.W. and Pollard, Tom J. and Shen, 
 Lu and Lehman, Li-wei H. and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, 
 Peter and Anthony Celi, Leo and Mark, Roger G. and et al.}, 
 year={2016}} 
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
MIMIC-III (‘Medical Information Mart for Intensive Care’) is a large, single-center database 
comprising information relating to patients admitted to critical care units at a large tertiary care hospital. 
Data includes vital signs, medications, laboratory measurements, 
observations and notes charted by care providers, fluid balance, procedure codes, 
diagnostic codes, imaging reports, hospital length of stay, survival data, and more.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://physionet.org/content/mimiciii-demo/1.4/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "https://physionet.org/content/mimiciii-demo/view-license/1.4/"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class MIMICDataset(datasets.GeneratorBasedBuilder):
    """MIMIC-III dataset"""

    VERSION = datasets.Version("1.4.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="minimum", version=VERSION, description="This is the bare minimum dataset"),    
    ]

    DEFAULT_CONFIG_NAME = "minimum"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "minimum":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "DOB": datasets.Value("string"),
                    "SEX": datasets.Value("string"),
                    "ADMITTIME": datasets.Value("string"),
                    "DISCHTIME": datasets.Value("string"),
                    "TEXT": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "DOB": datasets.Value("string"),
        #             "SEX": datasets.Value("string"),
        #             "ADMITTIME": datasets.Value("string"),
        #             "DISCHTIME": datasets.Value("string"),
        #             "TEXT": datasets.Value("string"),
        #             # These are the features of your dataset like images, labels ...
        #         }
        #    )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)

        data_dir = "./dataset/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.csv"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.csv"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            csvreader = csv.reader(f, delimiter=",")
            for key, row in enumerate(csvreader):
                if key == 0:
                    continue
                if self.config.name == "minimum":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "DOB": "" if split == "test" else row[0],
                        "SEX": "" if split == "test" else row[1],
                        "ADMITTIME": "" if split == "test" else row[2],
                        "DISCHTIME": "" if split == "test" else row[3],
                        "TEXT": row[4],
                    }
                # else:
                #     yield key, {
                #         "sentence": data["sentence"],
                #         "option2": data["option2"],
                #         "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                #     }