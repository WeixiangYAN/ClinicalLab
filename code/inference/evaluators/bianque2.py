import torch

from transformers import AutoTokenizer, AutoModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from evaluators.evaluator import Evaluator


class BianQue2Evaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, cache_dir=None, do_sample=False):
        super(BianQue2Evaluator, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            device_map='auto',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model = self.model.eval()
        self.do_sample = do_sample
        print(f'Memory footprint: {self.model.get_memory_footprint() / 1e6:.2f} MB')

    def format_prompt(self, prompt):
        return f'病人：{prompt}\n医生：'

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_text(self, prompt):
        prompt = self.format_prompt(prompt)
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            do_sample=self.do_sample
        )

        return response.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def count_tokens(self, prompt):
        return len(self.tokenizer(prompt)['input_ids'])
