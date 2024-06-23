import os

from sparkapi.config import SparkConfig
from sparkapi.core.chat.api import SparkAPI as ChatAPI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from evaluators.evaluator import Evaluator


class Spark3Evaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, app_id, api_secret, api_key, max_tokens=8192, temperature=0, top_k=1):
        super(Spark3Evaluator, self).__init__()

        os.environ['SPARK_APP_ID'] = app_id
        os.environ['SPARK_API_SECRET'] = api_secret
        os.environ['SPARK_API_KEY'] = api_key
        os.environ['SPARK_API_MODEL'] = pretrained_model_name_or_path
        os.environ['SPARK_CHAT_MAX_TOKENS'] = str(max_tokens)
        os.environ['SPARK_CHAT_TEMPERATURE'] = str(temperature)
        os.environ['SPARK_CHAT_TOP_K'] = str(top_k)

    def format_prompt(self, prompt):
        return prompt

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_text(self, prompt):
        prompt = self.format_prompt(prompt)
        api = ChatAPI(**SparkConfig().model_dump())
        response = api.get_completion(prompt)

        return ''.join(response).strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def count_tokens(self, prompt):
        return round((len(prompt) * 2) / 3)
