# Large Language Model Training and Fine-Tuning
Source code to train and fine-tune LLMs using TRL SFTTrainer and Deepspeed

## Setup your environment
The configurations including channels and packages to use are all specified in `env.yml`. The below command will create the required conda env. The `--prefix` in the command below points to the location of the new conda env. When you run the command below, make sure you replace `[USERNAME]` with your username on the server and `[ENV_NAME]` with the name of your conda env.

    conda env create --prefix /rep/[USERNAME]/envs/[ENV_NAME] -f environment.yml

## Generate dataset for training
I used the [CIDAR QA dataset](https://huggingface.co/datasets/arbml/CIDAR) to generate a small dataset to test LLM training.

To generate a sample of the CIDAR dataset, simply run the command below, the output data path `/path/to/your/data/` should be used as input to the training script.

    python /Users/mkhalilia/src/github/LLMTraining/src/blm/cli/process.py
        --output_path /path/to/your/data/ 
        --n 500

## Multi-GPU Training using Deepspeed
To train the LLM using multi-GPUs using Deepseed use the following command. Before you run the command below, some model like Llama require Huggingface token and you need to request access to these models. Some model can be used without a token, like Phi and Mistral. 

To obtain a Huggingface token, create a Huggingface account, go to settings, then access tokens.

In the training command below, you need to change the following values:
* `/path/to/src/blm/cli/train.py` => Update that the path to the training script
* `[GET_YOUR_TOKEN_FROM_HUGGINGFACE]` => Replace with your Huggingface token
* `/path/to/output/dir` => Path to model output
* `/path/to/src/blm/config/deepspeed_zero3.json` => Path to Deepspeed config 


    deepspeed --num_nodes=1 \
        /path/to/src/blm/cli/train.py 
        --model_name_or_path meta-llama/Llama-3.2-1B 
        --quantize False 
        --token [GET_YOUR_TOKEN_FROM_HUGGINGFACE] 
        --data_path /rep/mkhalilia/data/101/train 
        --max_seq_length 2048 
        --num_train_epochs 4 
        --per_device_train_batch_size 1 
        --gradient_checkpointing True 
        --output_dir /path/to/output/dir 
        --learning_rate 2e-4 
        --gradient_accumulation_steps 8 
        --bf16 True 
        --tf32 False 
        --logging_strategy steps 
        --save_strategy steps 
        --deepspeed /path/to/src/blm/config/deepspeed_zero3.json  
        --logging_steps 100 
        --save_steps 100 
        --lora_r 64 
        --lora_alpha 32 
        --per_device_eval_batch_size 1 
        --eval_strategy steps 
        --eval_accumulation_steps 1 

## Update PYTHONPATH
You may encounter an error `blm` module not found. When you run the command on the GPU, you need to update the PYTHONPATH to include the path to the `blm` package. To do so, run the following command (`blm` should be located in /path/to/src):

    export PYTHONPATH="${PYTHONPATH}:/path/to/src"
