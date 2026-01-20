import asyncio
import json
from typing import Any, AsyncIterator, Optional

from aiomqtt import Client


class MqttClient:
    def __init__(self, host: str, port: int = 1883, client_id: Optional[str] = None) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self._client: Optional[Client] = None

    async def __aenter__(self) -> "MqttClient":
        self._client = Client(self.host, self.port, identifier=self.client_id)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def publish_json(self, topic: str, payload: Any) -> None:
        if not self._client:
            raise RuntimeError("MQTT client not connected")
        data = json.dumps(payload)
        await self._client.publish(topic, data)

    async def subscribe_json(self, topic: str) -> AsyncIterator[tuple[str, Any]]:
        if not self._client:
            raise RuntimeError("MQTT client not connected")
        await self._client.subscribe(topic)
        async for msg in self._client.messages:
            try:
                payload = json.loads(msg.payload.decode())
            except json.JSONDecodeError:
                payload = None
            yield msg.topic.value, payload


async def keepalive_forever() -> None:
    while True:
        await asyncio.sleep(3600)
