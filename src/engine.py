from dataclasses import dataclass
import os
import re
import random
from typing import Callable, List
from loguru import logger
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

default_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    max_tokens=8192,
    stop=[
        "</answer>",
        "User:",
        "Human:",
        "Assistant:",
        "<|im_end|>",
        "<|endoftext|>",
    ],
)


def extract_code(full_output: str) -> str:
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "EXTRACTED CODE NOT FOUND"
    return code_output


@dataclass
class UinferOutput:
    prompt: str
    sample_num: int
    outputs: List[str]
    extracted_outputs: List[str] | None


####### vllm inference #######
class VLLMInferenceEngine:
    def __init__(
        self,
        pretrained_model,
        gpu_groups,
    ):
        self.pretrained_model = pretrained_model
        self.gpu_groups = gpu_groups
        self.task_queues = []
        self.result_queues = []
        self.processes = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def worker_fn(self, gpu_ids, task_queue, result_queue):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        llm = LLM(
            model=self.pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.85,
            max_model_len=self.max_model_len,
        )
        while True:
            task = task_queue.get()
            if task == "STOP":
                print("Stopping worker...")
                break
            task_id, prompts = task
            outputs = llm.generate(prompts, self.sampling_params)
            result_texts = [out.outputs[0].text for out in outputs]
            result_queue.put((task_id, result_texts))

    def start_workers(self):
        self.task_queues = []
        self.result_queues = []
        self.processes = []
        logger.info("Starting VLLM inference workers...")
        for i, gpu_ids in enumerate(self.gpu_groups):
            logger.info(f"Starting worker {i} on GPUs {gpu_ids}")
            task_q = mp.Queue()
            result_q = mp.Queue()
            p = mp.Process(
                target=VLLMInferenceEngine.worker_fn,
                args=(
                    self,
                    gpu_ids,
                    task_q,
                    result_q,
                ),
            )
            p.start()
            self.task_queues.append(task_q)
            self.result_queues.append(result_q)
            self.processes.append(p)

    def submit_prompt_set(self, prompt_sets):
        for i, prompts in enumerate(prompt_sets):
            self.task_queues[i].put((i, prompts))

    def collect_results(self, num_sets):
        results = [None] * num_sets
        for q in self.result_queues:
            task_id, result = q.get()
            results[task_id] = result
        return results

    def stop_workers(self):
        for q in self.task_queues:
            q.put("STOP")
        for p in self.processes:
            p.join()

    def split_prompts(self, prompts, n):
        k, m = divmod(len(prompts), n)
        return [
            prompts[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
        ]

    @staticmethod
    def get_token_lengths(strings, tokenizer):
        return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

    def generate_results(self, all_prompts):
        prompt_sets = self.split_prompts(all_prompts, len(self.gpu_groups))
        self.submit_prompt_set(prompt_sets)
        results = self.collect_results(len(prompt_sets))
        result_list = []
        for result_set in results:
            for r in result_set:
                result_list.append(r)
        return result_list

    def run(
        self,
        prompts: List[str],
        sample_num: int = 1,
        parse_output: Callable = extract_code,
        sampling_params: SamplingParams = default_sampling_params,
        max_model_len: int = 20000,
    ) -> List[UinferOutput]:
        self.sampling_params = sampling_params
        self.max_model_len = max_model_len
        # Start workers and tokenizer
        self.start_workers()

        all_prompts = [p for p in prompts for _ in range(sample_num)]
        N = len(all_prompts)

        indices = list(range(N))
        shuffled_idx = indices[:]
        random.shuffle(shuffled_idx)
        shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
        logger.info(f"Prompts: {len(prompts)}, Sample number: {sample_num}")
        logger.info(f"Starting inference job.., {N} prompts to inference.")
        logger.info(f"Example prompt: {shuffled_prompts[0]}")
        # Generate
        shuffled_outputs = self.generate_results(shuffled_prompts)
        restored_outputs = [None] * N
        for out, idx in zip(shuffled_outputs, shuffled_idx):
            restored_outputs[idx] = out

        ret = []
        # Group restored_outputs into nested lists, each of size sample_num
        grouped_outputs = [
            restored_outputs[i : i + sample_num] for i in range(0, N, sample_num)
        ]

        for i, go in enumerate(grouped_outputs):
            outputs = go
            extracted_outputs = [parse_output(out) for out in go]
            ret.append(
                UinferOutput(
                    prompt=prompts[i],
                    sample_num=sample_num,
                    outputs=outputs,
                    extracted_outputs=extracted_outputs,
                )
            )

        # self.stop_workers(self.task_queues, self.processes)
        return ret
