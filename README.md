# Xtemporal-reasoning
Project for the seminar "Event Processing" (M.A. Computational Linguistics at Heidelberg University), examining temporal reasoning of large language models (LLM).
This study experiments with the [TimeLlama](https://github.com/chenhan97/TimeLlama) model on the [TRAM](https://github.com/EternityYW/TRAM-Benchmark/tree/main) benchmark dataset.


## Installation

Set up a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```


Download the dataset:

`bash download-data.sh`


In case you are encountering access/login issues with Huggingface, you might have to be logged in (huggingface-cli login) and [request llama access](https://llama.meta.com/llama-downloads) with the same email address as your Huggingface account.



## Usage

To reproduce the experiments, run `python src/run.py` or `sbatch run.sh` on a GPU (insert your email address [here](https://github.com/LydiaKoerber/Xtemporal-reasoning/blob/main/run.sh#L5)).
To reproduce the analyses in [analysis.ipynb](https://github.com/LydiaKoerber/Xtemporal-reasoning/blob/main/analysis.ipynb), run the following to unzip the output file:

```
cd output
unzip outputs-nc.zip
rm outputs-nc.zip
```

## References
* model: TimeLlama [Paper](https://arxiv.org/abs/2310.01074) [GitHub](https://github.com/chenhan97/TimeLlama) [HuggingFace](https://huggingface.co/chrisyuan45/TimeLlama-7b)
* dataset: TRAM [Paper](https://arxiv.org/abs/2310.00835) [GitHub](https://github.com/EternityYW/TRAM-Benchmark/tree/main)

## Author
Lydia Körber
