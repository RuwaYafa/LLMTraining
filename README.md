# Large Language Model Training and Fine-Tuning
Source code to train and fine-tune LLMs using TRL SFTTrainer and Deepspeed

## Setup your environment
The configurations including channels and packages to use are all specified in `env.yml`. The below command will create the required conda env. The `--prefix` in the command below points to the location of the new conda env. When you run the command below, make sure you replace `[USERNAME]` with your username on the server and `[ENV_NAME]` with the name of your conda env.

    conda env create --prefix /rep/[USERNAME]/envs/[ENV_NAME] -f env.yml

## Generate dataset for training
I used the [101 Arabic Billion dataset](https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset) to generate a small dataset to test LLM training.

You can download one of the files from the 101 Arabic Billion dataset, from which you can generate small dataset. Use the command below to generate a test dataset.

    python /Users/mkhalilia/src/github/LLMTraining/src/blm/cli/train.py
        --input_path /path/to/downloaded/101/data.jsonl 
        --output_path /path/to/your/data/ 
        --n 1000

## Multi-GPU Training using Deepspeed
To train the LLM using multi-GPUs using Deepseed use the following command.

    deepspeed --num_nodes=1 \
        /rep/mkhalilia/src/blm/cli/train.py 
        --model_name_or_path meta-llama/Llama-3.2-1B 
        --quantize False 
        --token [GET_YOUR_TOKEN_FROM_HUGGINGFACE] 
        --data_path /rep/mkhalilia/data/101/train 
        --max_seq_length 2048 
        --num_train_epochs 4 
        --per_device_train_batch_size 1 
        --gradient_checkpointing True 
        --output_dir /rep/mkhalilia/model 
        --learning_rate 2e-4 
        --gradient_accumulation_steps 8 
        --bf16 True 
        --tf32 False 
        --logging_strategy steps 
        --save_strategy steps 
        --deepspeed /rep/mkhalilia/src/blm/config/deepspeed_zero3.json  
        --logging_steps 100 
        --save_steps 100 
        --lora_r 64 
        --lora_alpha 32 
        --per_device_eval_batch_size 1 
        --eval_strategy steps 
        --eval_accumulation_steps 1 