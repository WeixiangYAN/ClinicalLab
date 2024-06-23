import google.generativeai as gemini

from google.api_core import retry

from evaluators.evaluator import Evaluator


class GeminiProEvaluator(Evaluator):

    def __init__(self, pretrained_model_name_or_path, api_key, temperature=0):
        super(GeminiProEvaluator, self).__init__()

        gemini.configure(
            api_key=api_key,
            transport='rest'
        )
        self.model = gemini.GenerativeModel(pretrained_model_name_or_path)
        self.temperature = temperature
        self.candidate_count = 1
        self.max_output_tokens = 2048

    def format_prompt(self, prompt):
        return prompt

    @retry.Retry()
    def generate_text(self, prompt):
        prompt = self.format_prompt(prompt)
        response = self.model.generate_content(
            contents=prompt,
            generation_config=gemini.types.GenerationConfig(
                temperature=self.temperature,
                candidate_count=self.candidate_count,
                max_output_tokens=self.max_output_tokens
            )
        )

        return response.text.strip() if response.text is not None else response.text

    @retry.Retry()
    def count_tokens(self, string):
        return gemini.count_message_tokens(prompt=string)['token_count']
