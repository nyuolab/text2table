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
"""Loading script for MIMIC3 dataset."""


import csv
from multiprocessing.sharedctypes import Value
import os
import sys

import datasets


# BibTeX citation
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

# description of the dataset
_DESCRIPTION = """\
MIMIC-III (‘Medical Information Mart for Intensive Care’) is a large, single-center database 
comprising information relating to patients admitted to critical care units at a large tertiary care hospital. 
Data includes vital signs, medications, laboratory measurements, 
observations and notes charted by care providers, fluid balance, procedure codes, 
diagnostic codes, imaging reports, hospital length of stay, survival data, and more.
"""

# a link to an official homepage for the dataset here
_HOMEPAGE = "https://physionet.org/content/mimiciii-demo/1.4/"

# the licence for the dataset here if you can find it
_LICENSE = "https://physionet.org/content/mimiciii-demo/view-license/1.4/"


class MIMICDataset(datasets.GeneratorBasedBuilder):
    """MIMIC-III dataset"""

    VERSION = datasets.Version("1.4.0")

    # This is a list of multiple configurations for the dataset.
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="minimum", version=VERSION, description="This is the bare minimum dataset"),
        datasets.BuilderConfig(name="full", version=VERSION, description="This is the full dataset"),  
        datasets.BuilderConfig(name="dev", version=VERSION, description="This is a part of the full dataset for development"),
    ]

    #This is the default configuration
    DEFAULT_CONFIG_NAME = "full"

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "minimum":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            # These are the features of your dataset
            features = datasets.Features(
                {
                    "DOB": datasets.Value("string"),
                    "SEX": datasets.Value("string"),
                    "ADMITTIME": datasets.Value("string"),
                    "ICD9": datasets.Value("string"),
                    "HEADER": datasets.Value("string"),
                    "TABLE": datasets.Value("string"),
                    "TEXT": datasets.Value("string"),
                }
            )

        elif self.config.name == "full": # For the full dataset
            features = datasets.Features(
                {
                    "category": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        
        elif self.config.name == "dev": # For the development dataset
            features = datasets.Features(
                {
                    "category": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # This method is tasked with defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), 
        # the configuration selected by the user is in self.config.name

        # the original template of this method is also tasked with downloading and extracting data from URLs. 
        # However, since we work with our data locally, we don't need this functionality. 
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS

        if self.config.name == "minimum": # For the minimum dataset
            data_dir = "/gpfs/data/oermannlab/project_data/text2table/minimum_re_adtime"
        elif self.config.name == "full": # For the full dataset
            data_dir = "/gpfs/data/oermannlab/project_data/text2table/complete/train_test_data"
        elif self.config.name == "dev": # For the development dataset
            data_dir = "/gpfs/data/oermannlab/project_data/text2table/complete/dev_data"
        
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
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        if self.config.name == "minimum":
            with open(filepath, encoding="utf-8") as f:
                header_seq = "" #the sequence representation of the column headers
                csvreader = csv.reader(f, delimiter=",")
                for key, row in enumerate(csvreader):
                    if key == 0:
                        header_seq = " <COL> ".join([x for x in row if x != row[4]]) + " <ROW> "
                        continue
                    #the sequence representation of the nonheader cells
                    nonheader_seq = " <COL> ".join([x for x in row if x != row[4]])
                    # Yields examples as (key, example) tuples
                    yield (key - 1), {
                        "SEX": row[0],
                        "DOB": row[1],
                        "ADMITTIME": row[2],
                        "ICD9": row[3],
                        "HEADER": header_seq,
                        "TABLE": header_seq + nonheader_seq,
                        "TEXT": row[4],
                    }
        
        elif self.config.name == "full":
            csv.field_size_limit(sys.maxsize)
            with open(filepath, "r") as f:
                csvreader = csv.reader(f, dialect="excel")
                for key, row in enumerate(csvreader):
                    if key == 0:
                        continue
                    # The special token that we prepend the text with and also feed to the decoder
                    category_token = "<" + row[2] + ">"
                    # Yields examples as (key, example) tuples
                    yield (key - 1), {
                        "category": category_token,
                        "label": row[3],
                        # The list of texts that exclude empty strings
                        "text": " <text-sep> ".join([" ".join([category_token, x]) for x in row[4:] if x is not None and x != ""]),
                    }
        
        elif self.config.name == "dev":
            csv.field_size_limit(sys.maxsize)
            with open(filepath, "r") as f:
                csvreader = csv.reader(f, dialect="excel")
                for key, row in enumerate(csvreader):
                    if key == 0:
                        continue
                    # The special token that we prepend the text with and also feed to the decoder
                    category_token = "<" + row[2] + ">"
                    # Yields examples as (key, example) tuples
                    yield (key - 1), {
                        "category": category_token,
                        "label": row[3],
                        # The list of texts that exclude empty strings
                        "text": " <text-sep> ".join([" ".join([category_token, x]) for x in row[4:] if x is not None and x != ""]),
                    }
                    