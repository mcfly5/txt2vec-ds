from blocks import Processor
from loguru import logger
from requests import Session  # type: ignore
from blocks.connectors.http.events import OutputUrl

from config import Configuration
from src.events import ResponseEvent
from src.utils.utils import Measures, ErrorCodes, metric_sender

class Callback(Processor):
    def __init__(self, conf: Configuration) -> None:
        self.conf = conf

        self.endpoint = OutputUrl(
            url=conf.backend_callback_url,
            event=ResponseEvent,
            method='POST',
            headers={'Content-Type': 'application/json'},
        )

        self.session = Session()

    def __call__(self, response_event: ResponseEvent) -> None:
        try:
            logger.info('Notify backend...')
            logger.info(f'Callback to {self.endpoint.url} with data: {response_event.model_dump()}')
            data = response_event.model_dump()
            headers = {**self.session.headers, **self.endpoint.headers}
            cookies = {**self.session.cookies.get_dict(), **self.endpoint.cookies}
            method = getattr(self.session, self.endpoint.method.lower())
            response = method(
                self.endpoint.url, json=data, headers=headers, cookies=cookies,
                timeout=self.conf.backend_callback_timeout,
            )
            # Treat 4xx/5xx as a failed callback; the body may not be JSON, so log it as text.
            response.raise_for_status()
            logger.info(f'Backend response: {response.status_code} {response.text}')
        except Exception as err:
            logger.error(f'Error in callback: {err}')
            metric_sender.send_metric(
                name=Measures.error_code,
                value=ErrorCodes.BACKEND_ERROR)

    def close(self) -> None:
        self.session.close()
