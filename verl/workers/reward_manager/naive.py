# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import inspect
import time

from tqdm.asyncio import tqdm

from verl import DataProto
from verl.workers.reward_manager.base import BaseRewardManager


class NaiveRewardManager(BaseRewardManager):
    async def async_compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        semaphore = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency is not None else None
        min_interval = 1.0 / self.qps if self.qps > 0 else None
        lock = asyncio.Lock()
        last_called = 0.0

        async def throttled_call(data_item):
            nonlocal last_called
            if semaphore:
                await semaphore.acquire()
            try:
                if min_interval:
                    async with lock:
                        now = time.monotonic()
                        wait = min_interval - (now - last_called)
                        if wait > 0:
                            await asyncio.sleep(wait)
                        last_called = time.monotonic()
                return await asyncio.wait_for(
                    self.user_defined_compute_scores(data_sources=data_item.non_tensor_batch["data_sources"],
                                                     solution_strs=data_item.non_tensor_batch["solution_strs"],
                                                     ground_truths=data_item.non_tensor_batch["ground_truths"],
                                                     extra_infos=data_item.non_tensor_batch["extra_infos"]),
                    self.timeout)
            finally:
                if semaphore:
                    semaphore.release()

        return await tqdm.gather(*[throttled_call(reward_data[i]) for i in range(len(reward_data))], desc="Computing rewards...")

    def sync_compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        return [self.user_defined_compute_scores(
            data_sources=reward_data.non_tensor_batch["data_sources"][i],
            solution_strs=reward_data.non_tensor_batch["solution_strs"][i],
            ground_truths=reward_data.non_tensor_batch["ground_truths"][i],
            extra_infos=reward_data.non_tensor_batch["extra_infos"][i],
        ) for i in range(len(reward_data))]

    def compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        if inspect.iscoroutinefunction(self.user_defined_compute_scores):
            return asyncio.run(self.async_compute_scores(reward_data))
        else:
            return self.sync_compute_scores(reward_data)
