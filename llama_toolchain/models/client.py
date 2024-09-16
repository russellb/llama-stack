# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from pathlib import Path

from typing import Any, Dict, List, Optional

import fire
import httpx

from llama_toolchain.core.datatypes import RemoteProviderConfig
from termcolor import cprint

from .api import *  # noqa: F403


class ModelsClient(Models):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_models(self) -> List[ModelSpec]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models/list",
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return ModelsListResponse(**response.json())


async def run_main(host: str, port: int, stream: bool):
    client = ModelsClient(f"http://{host}:{port}")

    response = await client.list_models()
    cprint(response, "green")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
