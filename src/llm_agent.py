from loguru import logger
# Drop-in replacement for openai.OpenAI that records each call as a Langfuse generation.
from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from config import Configuration

class LLMAgent():
    def __init__(self, conf: Configuration) -> None:
        self.conf = conf

        self.client = OpenAI(
            base_url=conf.llm_base_url,
            api_key=conf.llm_api_key,
        )

    def call(self, question: str, system: str = '') -> ChatCompletionMessage:
        if not system:
            system = self.conf.llm_system_prompt
        try:
            completion = self.client.chat.completions.create(
                name='llm',
                model=self.conf.llm_model_name,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': question}
                ]
            )
        except Exception as e:
            logger.error(f'Error of LLM call: {str(e)}')
            raise

        return completion.choices[0].message
