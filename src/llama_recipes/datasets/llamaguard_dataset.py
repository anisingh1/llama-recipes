# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from datasets import load_dataset
from llama_recipes.data.llama_guard.finetuning_data_formatter import TrainingExample, Guidelines, Category, LlamaGuardPromptConfigs, LlamaGuardGenerationConfigs, AugmentationConfigs, FormatterConfigs, create_formatted_finetuning_examples


llamaguard_prompt="""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{guidelines}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for User in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""


categories = {
    "Violence and Hate": {
        "code": "O1",
        "description": """Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence."""
    },
    "Sexual Content": {
        "code": "O2",
        "description": """Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health."""
    },
    "Criminal Planning": {
        "code": "O3",
        "description": """Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems."""
    },
    "Guns and Illegal Weapons": {
        "code": "O4",
        "description": """Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership."""
    },
    "Regulated or Controlled Substances": {
        "code": "O5",
        "description": """Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation."""
    },
    "Self-Harm": {
        "code": "O6",
        "description": """Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves."""
    }
}


def intializeConfig():
    llamaguard_categories = []
    for key, value in categories.items():
        llamaguard_categories.append(Category(name=key, description=value["description"]))

    guidelines = Guidelines(categories=llamaguard_categories, category_code_prefix="O")

    # Setting up configs
    llama_guard_prompt_configs = LlamaGuardPromptConfigs(
        instructions_format_string=llamaguard_prompt,
        should_include_category_descriptions=True,
        should_shuffle_category_codes=True
    )

    llama_guard_generation_configs = LlamaGuardGenerationConfigs(
        should_list_violated_codes=True
    )

    augmentation_configs = AugmentationConfigs(
        should_add_examples_with_dropped_nonviolated_prompt_categories=True,
        should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True
    )

    formatter_configs = FormatterConfigs(
        guidelines=guidelines,
        llama_guard_prompt_configs=llama_guard_prompt_configs,
        llama_guard_generation_configs=llama_guard_generation_configs,
        augmentation_configs=augmentation_configs,
        random_seed=42
    )
    return formatter_configs


def prepareData(dataset, formatter_configs):
    training_examples = []
    for data in dataset:
        label = 'safe'
        category_codes = []
        if data["llamaguard offensive"] == True:
            label = 'unsafe'
            if data["llamaguard reasons"] != None:
                category = str(data["llamaguard reasons"]).split("|")
                for item in category:
                    item = item.strip()
                    category_codes.append(categories[item]["code"])
            else:
                print("Llamaguard data error: category not found for: " + data["text"])

        training_examples.append(
            TrainingExample(
                prompt=data["text"],
                response="N/A",
                violated_category_codes=category_codes,
                label=label
            )
        )

    # Call the create_formatted_finetuning_examples function
    formatted_examples = create_formatted_finetuning_examples(
        training_examples, formatter_configs)

    return formatted_examples


class LlamaguardDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.generator_config = intializeConfig()
        self.data_file_path: str = (
            dataset_config.train_data_path if partition == "train" else dataset_config.val_data_path
        )
        self.max_words: int = dataset_config.context_size
        self.tokenizer: LlamaTokenizer = tokenizer
        if self.data_file_path.endswith('csv'):
            try:
                dataset = load_dataset(
                    "csv",
                    data_files={partition: [self.data_file_path]},
                    delimiter=",",
                )[partition]
            except Exception as e:
                print("Loading of llamaguard dataset failed!")
                raise e
        else:
            raise ("Unknown data file format")
        
        self.data = prepareData(dataset, self.generator_config)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        input = self.data[index]["input"]
        output = self.data[index]["output"]
        input_ids = self.tokenizer.encode(self.tokenizer.bos_token + input, add_special_tokens=False)
        label_ids = self.tokenizer.encode(output + self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": input_ids + label_ids,
            "attention_mask": [1] * len(input_ids + label_ids),
            "labels": [-100] * len(input_ids) + label_ids
        }

        return sample
