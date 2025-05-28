import asyncio
from enum import Enum
from typing import Callable, Optional
import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def compute_score_with_throttle(
        evaluation_func, data_source, solution_str, ground_truth,
        extra_info=None,
        timeout: float = None,
        qps: int = None,
        max_concurrent_tasks: int = None
) -> float:
    """
    Computes a single score asynchronously, with optional rate limiting, concurrency limiting, and timeout.
    """
    interval = (1 / qps) if qps else 0
    semaphore = asyncio.Semaphore(max_concurrent_tasks) if max_concurrent_tasks else None

    async def _scoring():
        try:
            if interval > 0:
                await asyncio.sleep(interval)

            coro = evaluation_func(data_source, solution_str, ground_truth, extra_info)

            if timeout:
                return await asyncio.wait_for(coro, timeout=timeout)
            else:
                return await coro
        except asyncio.TimeoutError:
            print(f"[Timeout] Task timeout")
            return None
        except Exception as e:
            print(f"[Error] Task failed: {e}")
            return None

    if semaphore:
        async with semaphore:
            result = await _scoring()
    else:
        result = await _scoring()

    return result


class BaseRewardManager:
    """
        BaseRewardManager is the base class for all reward managers.
        It implements a general asynchronous scoring pipeline that performs batched reward evaluation
        using asyncio concurrency and supports throttling and concurrency control.

        Key Features:
        - Asynchronous scoring via asyncio coroutines.
        - Throttled request dispatch using configurable QPS (queries per second).
        - Bounded parallel execution using asyncio.Semaphore to limit max concurrent tasks.
        - Designed to be subclassed via overriding `calculate_reward(data)`, `post_process(data, scores, return_dict)` and `__call__(data, return_dict)`

        Default Parameters (can be customized in subclasses):
        - Timeout(time limit when calculating a single score. If not specified, task will wait until it completes): None
        - QPS (rate limit): None
        - Max Concurrent Tasks: None

        Subclasses must implement:
        - `calculate_reward(data_item)`: computes and returns the reward for a single sample.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer, num_examine: int,
                 compute_score: Optional[Callable] = None, reward_fn_key: str = "data_source", timeout: float = None,
                 qps: int = None, max_concurrent_tasks: int = None, **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.timeout = timeout
        self.qps = qps
        self.max_concurrent_tasks = max_concurrent_tasks

    def calculate_reward(self, data: DataProto):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        try:
            async def run_all():
                tasks = []
                for i in range(len(data)):
                    data_item = data[i]  # DataProtoItem
                    data_dict = self.retrive_data_for_processing(data_item)
                    task = compute_score_with_throttle(evaluation_func=self.compute_score,
                                                       data_source=data_dict["data_source"],
                                                       solution_str=self.get_response_str(data_item),
                                                       ground_truth=data_dict["ground_truth"],
                                                       extra_info=data_dict["extra_info"], timeout=self.timeout,
                                                       qps=self.qps,
                                                       max_concurrent_tasks=self.max_concurrent_tasks)
                    tasks.append(task)

                return await asyncio.gather(*tasks)

            scores = loop.run_until_complete(run_all())
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        finally:
            loop.close()
        return scores

    def post_process(self, data: DataProto, scores: list, return_dict: bool = False):
        """To be overridden by subclasses."""
        raise NotImplementedError

    def retrive_data_for_processing(self, data_item):
        data_dict = {}
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        data_dict["valid_response_length"] = valid_response_length
        data_dict["prompt_str"] = prompt_str
        data_dict["response_str"] = response_str
        data_dict["ground_truth"] = ground_truth
        data_dict["data_source"] = data_source
        data_dict["extra_info"] = extra_info
        return data_dict

    def get_response_str(self, data_item):
        data_dict = self.retrive_data_for_processing(data_item)
        return data_dict["prompt_str"]

    def __call__(self, data: DataProto, return_dict: bool = False):
        """To be overridden by subclasses."""
        raise NotImplementedError
