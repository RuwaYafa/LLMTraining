class Prompter:
    def __init__(self, tokenizer):
        templates = {
            "mistralai": {"instruction_template": "[INST]",
                          "response_template": "[/INST]"},
            "meta-llama": {"instruction_template": "<|start_header_id|>system<|end_header_id|>", 
                           "response_template": "<|start_header_id|>assistant<|end_header_id|>"},
            "microsoft": {"instruction_template": "<|system|>", 
                           "response_template": "<|assistant|>"},
        }

        self.tokenizer = tokenizer
        self.model_family = self.tokenizer.name_or_path.split("/")[0]
        self.instruction_template = templates[self.model_family]["instruction_template"]
        self.response_template = templates[self.model_family]["response_template"]

    def __call__(self, data=None, system=None, user=None):
        """Prepare input for model training or inference.
        Pass data to generate prompts for training
        Pass system and user to generate one time prompt for a specific model
        based on the model ID in the tokenizer.

        Args:
            data (DatasetDict): dataset that should contains prompt 
                                components (system, instructions, data and output)
                                Use this option when generating training data
            system (str): system prompt
            user (str): user prompt
        """
        if data:
            data["prompt"] = [self.for_training(example) for example in data['prompt']]
            return data
        elif system or user:
            prompt = self.for_langchain(system=system, user=user)
            return prompt
    
    def for_training(self, example):
        if self.model_family == "mistralai":
            user_with_data = example["instructions"].format(**example["template_variables"])
            user_with_inst = f"{example['system']}{user_with_data}"

            messages = [
                {"role": "user", "content": user_with_inst},
                {"role": "assistant", "content": example["output"]}
            ]
        else:
            user_with_data = example["instructions"].format(**example["template_variables"])

            messages = [
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": user_with_data},
                {"role": "assistant", "content": example["output"]}
            ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt

    def for_langchain(self, system=None, user=None):
        if self.model_family == "mistralai":
            messages = [
                {"role": "user", "content": system + user}
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt