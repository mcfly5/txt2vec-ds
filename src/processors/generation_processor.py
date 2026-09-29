from blocks import Processor
from langfuse.types import TraceContext
from loguru import logger

from config import Configuration
from src.events import Documents, ResponseEvent
from src.prompts import prompt_gen
from src.llm_agent import LLMAgent
from src.tracing import init_langfuse

class Generation(Processor):
    def __init__(self, conf: Configuration) -> None:
        self.conf = conf
        self.langfuse = init_langfuse(conf)
        self.llm = LLMAgent(self.conf)

    def __call__(self, documents: Documents) -> ResponseEvent:
        # Join the trace started by DataGetter; without a trace id a new trace is started.
        trace_id = getattr(documents.event, 'trace_id', None)
        trace_context = TraceContext(trace_id=trace_id) if trace_id else None
        with self.langfuse.start_as_current_span(
            name='generation', trace_context=trace_context
        ) as span:
            response = self.generate(documents)
            span.update(output={'response': response.response, 'doc_array': response.doc_array})
            span.update_trace(output=response.response)
        return response

    def generate(self, documents: Documents) -> ResponseEvent:
        self.query = documents.query
        self.documents = documents.documents
        logger.info('Started generation')

        prompt = prompt_gen.format(
            query=self.query,
            documents='\n'.join(['Document #:' + str(i + 1) + '\n' + doc.page_content for i,
                                doc in enumerate(self.documents)]) if self.documents else ''
        )
        logger.debug(prompt)
        llm_response = self.llm.call(prompt)
        logger.info(llm_response)
        if llm_response:
            documents.event.response = llm_response.content
            if self.documents:
                documents.event.doc_array = [doc.metadata['source'] for doc in self.documents]

        response = ResponseEvent(
            conversation_id=documents.event.conversation_id,
            response=documents.event.response,
            doc_array=documents.event.doc_array
        )
        return response
