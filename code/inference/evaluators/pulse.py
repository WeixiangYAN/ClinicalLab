import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential

from evaluators.evaluator import Evaluator


class PULSEEvaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, cache_dir=None, do_sample=False, max_length=4096):
        super(PULSEEvaluator, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            device_map='auto',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model = self.model.eval()
        self.do_sample = do_sample
        self.max_length = max_length
        print(f'Memory footprint: {self.model.get_memory_footprint() / 1e6:.2f} MB')

    def format_prompt(self, prompt):
        return f'Instructions: You are Helper, a large language model full of intelligence. Respond conversationally.</s>User: {prompt}</s> Helper: '

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_text(self, prompt):
        prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            inputs['input_ids'],
            do_sample=self.do_sample,
            max_length=self.max_length,
            eos_token_id=self.tokenizer.convert_tokens_to_ids('</s>')
        ).to('cpu')
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response.split('Helper:')[-1].strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def count_tokens(self, prompt):
        return len(self.tokenizer(prompt)['input_ids'])
