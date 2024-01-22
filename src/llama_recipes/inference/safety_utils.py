# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os, requests, json
import torch
import warnings
from auditnlg.safety.exam import safety_scores
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

# Class for performing safety checks using AuditNLG library
class AuditNLGSensitiveTopics(object):
    def __init__(self):
        pass

    def __call__(self, output_text, **kwargs):            
        try:
            data = [{"output": output_text}]
            result = safety_scores(data=data, method="sensitive_topics")
            scores = result[1]["all_scores"][0]
            is_safe = scores["pred_class"] == "none"
            report = ""
            if not is_safe:
                categories = []
                for item in scores["class_scores"]:
                    for key, value in item.items():
                        if value >= 0.5:
                            categories.append(key)
                report = " | ".join(categories)

            return "AuditNLG", is_safe, report
        except Exception as e:
            return "AuditNLG", "", ""
    
    
class SalesforceSafetyChecker(object):
    def __init__(self):
        try:
            config = AutoConfig.from_pretrained("Salesforce/safety-flan-t5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/safety-flan-t5-base") 
            self.model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/safety-flan-t5-base", config=config)
            self.model.eval()
        except ImportError:
            raise Exception("Could not import Salesforce model")

    def __call__(self, output_text, **kwargs):
        try:
            prefix = "Is the <Text> field safe or unsafe?"
            input_ids = self.tokenizer(prefix + " <Text> " + output_text + " <Context> ", return_tensors="pt").input_ids
            
            if len(input_ids[0]) > 512:
                warnings.warn(
                    "Input length is > 512 token. Safety check result could be incorrect."
                )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=20,
                )
            
            is_safe = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).split(" ")[0] == "safe"    
                
            report = ""
            if not is_safe:
                true_false_ids = self.tokenizer("true false").input_ids[:2]
                keys = ["toxicity", "hate", "identity", "violence", "physical", "sexual", "profanity", "biased"]
                scores = {}
                for k, i in zip(keys, range(3,20,2)):
                    scores[k] = round(outputs.scores[i][0,true_false_ids].softmax(dim=0)[0].item(), 5)
                
                categories = []
                for key, value in scores.items():
                    if value >= 0.5:
                        categories.append(key)
                report = " | ".join(categories)

            return "Salesforce", is_safe, report
        
        except Exception as e:
            return "Salesforce", "", ""
        

    def get_total_length(self, data):
        prefix = "Is the <Text> field safe or unsafe "
        input_sample = "<Text> {output} <Context> ".format(**data[0])

        return len(self.tokenizer(prefix + input_sample)["input_ids"])


# Class for performing safety checks using Azure Content Safety service
class AzureSaftyChecker(object):
    def __init__(self):
        try:
            key = os.environ["AZURE_CONTENT_SAFETY_KEY"]
            endpoint = os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"]
        except ImportError:
            raise Exception("Could not import required package azure-ai-contentsafety. Install with: pip install azure-ai-contentsafety")
        except KeyError:
            raise Exception("Environment variables not set. Please set AZURE_CONTENT_SAFETY_KEY and AZURE_CONTENT_SAFETY_ENDPOINT.")

        self.client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    def __call__(self, output_text, **kwargs):
        if len(output_text) > 1000:
            raise Exception("Input length to safety check is too long (>1000).")

        categories = [
            TextCategory.VIOLENCE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.HATE,
        ]

        request = AnalyzeTextOptions(text=output_text, categories=categories)
        try:
            response = self.client.analyze_text(request)
        except HttpResponseError as e:
            return "Azure", "", e

        try:
            hate_result = next(item for item in response.categories_analysis if item.category == TextCategory.HATE)
            self_harm_result = next(item for item in response.categories_analysis if item.category == TextCategory.SELF_HARM)
            sexual_result = next(item for item in response.categories_analysis if item.category == TextCategory.SEXUAL)
            violence_result = next(item for item in response.categories_analysis if item.category == TextCategory.VIOLENCE)

            report = ""
            if hate_result.severity > 0:
                report += hate_result.category
            if self_harm_result.severity > 0:
                report += self_harm_result.category
            if sexual_result.severity > 0:
                report += sexual_result.category
            if violence_result.severity > 0:
                report += violence_result.category

            is_safe = len(report) == 0 
            return "Azure", is_safe, report
        
        except Exception as e:
            return "Azure", "", ""


class LlamaGuardSafetyChecker(object):

    def __init__(self):
        model_id = "meta-llama/LlamaGuard-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        self.tags = {
            "O1": "Violence and Hate",
            "O2": "Sexual Content",
            "O3": "Criminal Planning",
            "O4": "Guns and Illegal Weapons",
            "O5": "Regulated or Controlled Substances",
            "O6": "Self-Harm"
        }

    def __call__(self, output_text, **kwargs):
        model_prompt = output_text.strip()
        chat = [{"role": "user", "content": model_prompt}]

        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
        prompt_len = input_ids.shape[-1]
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        
        splitted_result = result.split("\n")
        is_safe = splitted_result[0] == "safe"    

        if len(splitted_result) == 1:
            report = ""
        else:
            res = splitted_result[1].split(',')
            categories = []
            for item in res:
                categories.append(self.tags[item])
            report = " | ".join(categories)
        
        return "Llamaguard", is_safe, report
        

class GPTChecker(object):
    header = {}

    def __init__(self):
        self.header = {
            "api-key": os.environ["GPT_KEY"],
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.endpoint = os.environ["GPT_ENDPOINT"]


    def getMessage(self, text):
        context = "Provide a list of reasons with each reason described in around 10 words in JSON format along with their severity (categorized as High, Medium, and Low) for the last user-provided text input to be culturally offensive to the majority of people. Don't make any presumptions and only consider the user-provided text input for evaluation. Merge similar reasons together and provide the response within 150 words. Only provide a valid JSON in the response. Do not provide any explanations, examples or notes."
        conversation = [
            {
                "role":"system",
                "content":context
            },
            {
                "role":"user",
                "content":"Jesus Christ as a general of an Army"
            },
            {
                "role":"assistant",
                "content":"[{\"reason\": \"Disrespecting religious sentiments of Christians\",\"severity\": \"High\"},{\"reason\": \"Misrepresentation of Jesus' peaceful teachings\",\"severity\": \"High\"},{\"reason\": \"Promoting religious conflict in diverse India\",\"severity\": \"High\"},{\"reason\": \"Inappropriate portrayal of a religious figure\",\"severity\": \"Medium\"},{\"reason\": \"Ignorance of Indian cultural and religious values\",\"severity\": \"Medium\"}]"
            },
            {
                "role":"user",
                "content":"a baby playing with toys"
            },
            {
                "role":"assistant",
                "content":"[]"
            },
            {
                "role":"user",
                "content":text
            }
        ]
        obj = {
            'messages': conversation,
            'max_tokens': 256,
            'temperature': 0.1,
            'top_p': 0.95,
            'frequency_penalty': 0,
            'presence_penalty': 0
        }
        return obj


    def __call__(self, output_text, **kwargs):
        reqUrl = self.endpoint
        obj = self.getMessage(output_text)

        try:
            x = requests.post(reqUrl, data=json.dumps(obj), headers=self.header, timeout=30)
            x = json.loads(x.text)
            if 'choices' in x:
                if x['choices'][0]['message']['content'] != None:
                    x = x['choices'][0]['message']['content']
                    report = []
                    is_safe = True
                    reasons = json.loads(x)
                    if len(reasons) > 0:
                        for item in reasons:
                            if item['severity'] == 'High':
                                is_safe = False
                                report.append(item['reason'])
                            elif item['severity'] == 'Medium':
                                report.append(item['reason'])
                    report = " | ".join(report)
                else:
                    return "GPT4", "", ""
            elif 'message' in x:
                print("ERROR: " + output_text + " : " + x['message'])
                return "GPT4", "", ""
            return "GPT4", is_safe, report
        except Exception as e:
            print(e)
            return "GPT4", "", ""
        

class LLama2Checker(object):

    def __init__(self):
        self.header = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.endpoint = os.environ["LLAMA2_ENDPOINT"]
        self.model = os.environ["LLAMA2_MODEL"]


    def getMessage(self, text):
        context = "You are a content moderator. Find out if the following text written below CONTENT label may be offensive to majority of people. Provide your response in JSON format with a boolean field 'offensive' and an array field 'reasons' as JSON keys. If the text is not offensive keep the 'offensive' field as 'false' and 'reasons' field as '[]' otherwise make 'offensive' field as 'true' and provide list of reasons the content is offensive in the 'reasons' field in JSON with each reasons being less than 10 words. For example: if the input is '3d house icon with dark grey roof and israel flag ontop' then output should be '{\n\"offensive\": true,\n\"reasons\": [\"Potential political implications related to Israel-Palestine conflict\"]\n}. Please make note to provide only the JSON in reponse as shown in the example above. DO NOT provide explanation about your response."
        conversation = [
            {
                "role":"user",
                "content":context + "\n CONTENT \n" + text
            }
        ]
        obj = {
            'messages': conversation,
            'max_tokens': 256,
            'temperature': 0.1,
            'top_p': 0.95,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'model': self.model
        }
        return obj


    def __call__(self, output_text, **kwargs):
        reqUrl = self.endpoint
        obj = self.getMessage(output_text)

        try:
            x = requests.post(reqUrl, data=json.dumps(obj), headers=self.header, timeout=30)
            x = json.loads(x.text)
            is_safe = True
            if 'choices' in x:
                if x['choices'][0]['message']['content'] != None:
                    x = x['choices'][0]['message']['content']
                    report = []
                    output = json.loads(x)
                    if len(output['reasons']) > 0:
                        is_safe = False
                        for item in output['reasons']:
                            report.append(item)
                    report = " | ".join(report)
                else:
                    return "LLama2", "", ""
            elif 'message' in x:
                return "LLama2", "", ""
            return "LLama2", is_safe, report
        except Exception as e:
            print(e)
            return "LLama2", "", ""
        

# Function to load the PeftModel for performance optimization
# Function to determine which safety checker to use based on the options selected
def get_safety_checker(enable_azure_content_safety,
                       enable_sensitive_topics,
                       enable_salesforce_content_safety,
                       enable_llamaguard_content_safety,
                       enable_gpt_safety,
                       enable_llama2_safety):
    safety_checker = []
    if enable_azure_content_safety:
        safety_checker.append(AzureSaftyChecker())
    if enable_sensitive_topics:
        safety_checker.append(AuditNLGSensitiveTopics())
    if enable_salesforce_content_safety:
        safety_checker.append(SalesforceSafetyChecker())
    if enable_llamaguard_content_safety:
        safety_checker.append(LlamaGuardSafetyChecker())
    if enable_gpt_safety:
        safety_checker.append(GPTChecker())
    if enable_llama2_safety:
        safety_checker.append(LLama2Checker())
    return safety_checker

