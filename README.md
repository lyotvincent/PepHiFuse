# PepHiFuse
Adaptive Fusion of Global and Local Representations for Neoantigen Retention Time Prediction through Hierarchical Sequence-Graph Hybridization

## Quickstart

### 1. Install all requirements.

Using Docker: Download and run the docker image. Ensure you have [Docker](https://docs.docker.com/desktop/install/ubuntu/) installed.
```shell
docker pull lyrmagical/pephifuse
docker run -it [--gpus all] --network host --name pephifuse lyrmagical/pephifuse /bin/bash
git clone https://github.com/lyotvincent/PepHiFuse
```
[--gpus all] requires the installation of [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit), which allows users to run GPU accelerated containers.

### 2. Set up third-party libraries and generate data.

Download [ProtT5-XL-UniRef50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) model and put it under the PepHiFuse folder.
```shell
mv prot_t5_xl_uniref50 PepHiFuse
cd PepHiFuse
```
An example input data is available in the `example` folder. The input data should have at least 2 columns `x` and `y` in a **`tab-delimited text format`**. The `x` column contains peptide sequences and `y` column contains retention time. The unit of retention time is **minute**. To generate the training data, you can run the script as shown below.
```shell
python scripts/generate_rt_data.py example/example.tsv example/train.txt --sample_ratio 0.8
```

### 3. Training a model.

You can launch a training by pointing to train and eval data:
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/run_language_modeling.py --model_type xlnet --output_dir results \
    --config_name configs/config_for_rt_task.json --tokenizer_name vocabs/rt_vocab.txt \
    --do_train --do_eval --learning_rate 5e-5 --num_train_epochs 120 --save_total_limit 5 \
    --save_steps 1000 --per_gpu_train_batch_size 32 --evaluate_during_training --eval_steps 100 \
    --eval_data_file example/eval.txt --train_data_file example/train.txt \
    --line_by_line --seed 42 --logging_steps 1000 --eval_accumulation_steps 1 \
    --training_config_path training_configs/rt_alternated_cc.json --per_gpu_eval_batch_size 32
```

### 4. Evaluating a model.

To evaluate a model trained, run the following:
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/eval_language_modeling.py --output_dir results \
--eval_file example/eval.txt --eval_accumulation_steps 1 --param_path configs/config_for_rt_task_eval.json
```

## Contact

If you have any problems, just raise an issue in this repo.