class Prompter:
    def __init__(self, tokenizer):
        templates = {
            "mistralai": {"instruction_template": "[INST]",
                          "response_template": "[/INST]"},
            "meta-llama": {"instruction_template": "<|start_header_id|>system<|end_header_id|>", 
                           "response_template": "<|start_header_id|>assistant<|end_header_id|>"},
            "microsoft": {"instruction_template": "<|system|>", 
                           "response_template": "<|assistant|>"},
            "default": {"instruction_template": "###Instructions:\n\n",
                          "response_template": "###Assistant:\n\n"},
        }

        self.tokenizer = tokenizer
        self.model_family = self.tokenizer.name_or_path.split("/")[0]
        self.instruction_template = templates.get(self.model_family, templates["default"])["instruction_template"]
        self.response_template = templates.get(self.model_family, templates["default"])["response_template"]

    def __call__(self, data):
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
        data["prompt"] = [self._for_sft(messages) for messages in data["messages"]]
    
    def _for_sft(self, messages):
        """
        Convert the list of messages into a prompt by injecting the LLM special
        instruction and assistant tokens.
        :param messages: List[Dict] - list of user and assistant messages
        :return: str - prompt used as input to model training
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
