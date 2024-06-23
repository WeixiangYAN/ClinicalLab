import anthropic

from tenacity import retry, stop_after_attempt, wait_random_exponential

from evaluators.evaluator import Evaluator


class Claude3Evaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, api_key, temperature=0):
        super(Claude3Evaluator, self).__init__()

        self.client = anthropic.Anthropic(
            api_key=api_key
        )
        self.model = pretrained_model_name_or_path
        self.temperature = temperature
        self.max_tokens = 4096

    def format_prompt(self, prompt):
        return [
            {
                'role': 'user',
                'content': prompt
            }
        ]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_text(self, prompt):
        prompt = self.format_prompt(prompt)
        response = self.client.messages.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.content[0].text.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def count_tokens(self, prompt):
        return self.client.count_tokens(prompt)
