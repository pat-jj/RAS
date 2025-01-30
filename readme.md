# RAS: Enhanced Knowledge-Intensive LLM Generation with Iterative Retrieval-And-Structure

### Table of Contents
- [Environment Setup](#environment-setup)
- [Train Theme Classifier and Distribution Shifter](#train-theme-classifier-and-distribution-shifter)
  - [Train](#train)
  - [Test](#test)
- [Train Text-to-Triples Model](#train-text-to-triples-model)
- [Training Data (HotpotQA-SUBQ) Processing](#training-data-hotpotqa-subq-processing)
- [Train GraphLLM by Multi-task Learning](#train-graphllm-by-multi-task-learning-w-processed-training-data)
  - [Train](#train-1)
  - [Test](#test-1)
- [Knowledge Indexing](#knowledge-indexing-prepare-both-theme-and-dense-faiss-indexes)
- [Run Baselines](#run-baselines)
- [Run RAS](#run-ras)

---
### Environment Setup

Please follow the commands below (in exact order) to setup the environment.

```bash
# Create new env
conda create -n ras python=3.10 -y

# Activate it
conda activate ras

# Install PyTorch first, separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install torch-geometric

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html #(depends on your cuda version)

pip install transformers wandb tqdm peft accelerate bitsandbytes sentencepiece

```


### Train Theme Classifier and Distribution Shifter

First, download DBPedia-298 dataset from [here](https://www.kaggle.com/datasets/danofer/dbpedia-classes).

#### [Train]
```bash
cd classifier_shifter
sh doc_train.sh
sh shifter_train.sh # please process HotpotQA-SUBQ data (see below) before this
```

#### [Test]
```bash
cd classifier_shifter
python theme_predictor.py
```

### Train Text-to-Triples Model

First, download WikiOFGraph dataset from [here](https://drive.google.com/drive/folders/1FaEdfgmcjHixVacdZLFCus6HO-k2yrR5?usp=sharing).

```bash
cd text_to_triples
sh train.sh
```




### Training Data (HotpotQA-SUBQ) Processing
First, download hotpotqa training set from [here](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json).

Then, run the following commands to process the training data.

```bash
cd llm_training_data_process
# Process the hotpotqa data
python process_hotpot.py

# Generate subqueries for hotpotqa
python generate_subqueries.py

# identify questions that don't need both subqueries and retrieval
python training_data_gen_wo_ret_wo_subq.py

cd ../text_to_triples

# generate triples for hotpotqa docs
python generate.py

# process the data with graphs
cd ../llm_training_data_process
python a_planner_data_process.py
python a_1_hotpotqa_only.py
python b_answerer_data_process.py
```


### Train GraphLLM by Multi-task Learning (w/ processed training data)

#### [Train]
```bash
cd framework
sh train_planner.sh
sh train_answerer.sh
```

#### [Test] (w/ hotpotqa-subq validation data)
```bash
sh test_planner.sh
sh test_answerer.sh
```


### Knowledge Indexing (Prepare both Theme and Dense Faiss Indexes)

```bash
cd knowledge_indexing/theme
sh class_labeling.sh
sh convert.sh

cd ../dense
sh dense_index.sh
sh combine.sh
```

### Run Baselines

```bash
cd baselines
sh run.sh # (please see the arguments in the run.sh file to change the dataset, model, etc.)
```


### Run RAS

```bash
cd framework
sh run_ras.sh
```


#### Note: To run closed-source Sonnet-3.5 in either baselines' setting or RAS, please fill in the key information in the `claude_api_example.py` file, and rename it to `claude_api.py`, and put it under both baselines/ and framework/.

---