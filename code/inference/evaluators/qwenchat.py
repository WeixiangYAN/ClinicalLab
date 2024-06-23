import dashscope

from http import HTTPStatus
from tenacity import retry, stop_after_attempt, wait_random_exponential

from evaluators.evaluator import Evaluator


class QwenChatEvaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, api_key, temperature=0):
        super(QwenChatEvaluator, self).__init__()

        dashscope.api_key = api_key
        self.model = pretrained_model_name_or_path
        self.temperature = temperature

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
        response = dashscope.Generation.call(
            self.model,
            messages=prompt,
            temperature=self.temperature,
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content'].strip()
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (response.request_id, response.status_code, response.code, response.message))
            return ''

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def count_tokens(self, prompt):
        response = dashscope.Tokenization.call(
            'qwen-14b-chat',
            prompt=prompt,
        )
        if response.status_code == HTTPStatus.OK:
            return response['usage']['input_tokens']
        else:
            print('Failed request_id: %s, status_code: %s, code: %s, message:%s' % (response.request_id, response.status_code, response.code, response.message))
            return 0
