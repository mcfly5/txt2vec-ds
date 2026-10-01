from blocks import Processor
from langfuse.types import TraceContext
from loguru import logger

from config import Configuration
from src.events import Documents, ResponseErrorCode, ResponseEvent, ResponseStatus
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
            # Always produce a ResponseEvent: the callback must notify the backend on failures too.
            try:
                response = self.generate(documents)
            except Exception as err:
                logger.exception(f'Generation failed: {err}')
                documents.event.fail(ResponseErrorCode.INTERNAL_ERROR, err)
                response = self.build_response(documents)
            span.update(output=response.model_dump(exclude={'conversation_id', 'trace_id'}))
            if response.status == ResponseStatus.FAILED:
                span.update(level='ERROR', status_message=response.error_message)
            span.update_trace(output=response.response)
        return response

    def generate(self, documents: Documents) -> ResponseEvent:
        self.query = documents.query
        self.documents = documents.documents
        event = documents.event
        if event.status == ResponseStatus.FAILED:
            logger.warning(f'Skip generation, retrieval failed: {event.error_message}')
            return self.build_response(documents)
        if not self.documents:
            logger.info('No documents retrieved, skip generation')
            event.status = ResponseStatus.NO_DOCUMENTS
            return self.build_response(documents)
        logger.info('Started generation')

        prompt = prompt_gen.format(
            query=self.query,
            documents='\n'.join(['Document #:' + str(i + 1) + '\n' + doc.page_content for i,
                                doc in enumerate(self.documents)]) if self.documents else ''
        )
        logger.debug(prompt)
        try:
            llm_response = self.llm.call(prompt)
        except Exception as err:
            event.fail(ResponseErrorCode.LLM_ERROR, err)
            return self.build_response(documents)
        logger.info(llm_response)
        if not llm_response.content:
            event.fail(ResponseErrorCode.LLM_ERROR, 'Empty LLM response')
            return self.build_response(documents)
        event.response = llm_response.content
        event.doc_array = [doc.metadata['source'] for doc in self.documents]
        return self.build_response(documents)

    @staticmethod
    def build_response(documents: Documents) -> ResponseEvent:
        event = documents.event
        completed = event.status == ResponseStatus.COMPLETED
        return ResponseEvent(
            conversation_id=event.conversation_id,
            response=event.response if completed else '',
            doc_array=event.doc_array if completed else [],
            status=event.status,
            error_code=event.error_code,
            error_message=event.error_message,
            trace_id=event.trace_id,
        )
