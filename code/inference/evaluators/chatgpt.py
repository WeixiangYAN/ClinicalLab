import openai
import backoff
import tiktoken

from evaluators.evaluator import Evaluator


class ChatGPTEvaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, api_key, temperature=0):
        super(ChatGPTEvaluator, self).__init__()

        openai.api_key = api_key
        self.model = pretrained_model_name_or_path
        self.temperature = temperature

    def format_prompt(self, prompt):
        return [
            {
                'role': 'user',
                'content': prompt
            }
        ]

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def generate_text(self, prompt):
        prompt = self.format_prompt(prompt)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature
        )

        return response['choices'][0]['message']['content'].strip()

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def count_tokens(self, prompt):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(prompt))

        return num_tokens
