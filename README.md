# Intriguing Effect of the Correlation Prior on ICD-9 Code Assignment

> The Ninth Revision of the International Classification of Diseases (ICD-9) is 
> a complex coding system for classifying health conditions, which researchers have attempted to 
> automate using language models. However, the imbalanced distribution of ICD-9 codes leads to 
> poor performance, prompting exploration of using the correlation bias between codes to improve results. 
> While the correlation bias has the potential to enhance code assignment in specific cases, 
> it worsen the overall performance. This repo contains script for datasets, experiments, and figures in this paper.


## Structure and implementation

The `language_model` folder contains source code for all experiments and model implementation.
The `notebooks` folder contains figures and tables generated inside [Jupyter notebooks](http://jupyter.org/).
The `dataset` folder contains source code for creating the dataset used for experiments.
The `utils` folder contains source code for helper functions and utilities used for figures and experiments.
See the `README.md` files in each directory for a full description.


## Installation:

placeholder for installation

## Dependencies

```yaml

- dask==2022.5.2
- datasets==2.2.2
- deepspeed==0.8.2
- torch==2.0.0+cu118
- tqdm==4.64.0
- transformers==4.27.1
- scikit-learn==1.1.1

```
You need to install the version specified here for these packages.
Please see `requirements.txt` for the full list of packages. 




<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# Text2Table

