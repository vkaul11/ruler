import json
import math
import random
import string
from dataclasses import dataclass
from itertools import product
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def write_file(data_str, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data_str)


def _make_random_string(length: int) -> str:
    alphabet = string.ascii_letters
    return "".join(random.choice(alphabet) for _ in range(length))


@dataclass
class MultiHopSample:
    query: str
    matched_fragments_content: List[str]
    all_fragments_content: List[str]


class MultiHopEval:
    @staticmethod
    def make_one(
        n_chars_problem: int, num_queries: int, hops: int, hash_pair_str_length: int
    ) -> List["MultiHopSample"]:
        chars_per_hash_pair = ((hash_pair_str_length * 2 + 3)) * hops
        n_chains = max(1, math.ceil(n_chars_problem / chars_per_hash_pair))

        # Generate levels
        levels = MultiHopEval._make_levels(
            n=n_chains, hops=hops, string_length=hash_pair_str_length
        )

        # Build the full variable mapping
        variable_to_value = {}
        all_fragments_content = []
        for level in levels:
            variable_to_value.update(level)
            all_fragments_content.extend([f"{k} = {v}" for k, v in level.items()])

        # Generate queries and matched_fragments_content
        samples = []
        initial_variables = list(levels[0].keys())
        random.shuffle(initial_variables)
        selected_queries = initial_variables[:num_queries]

        for query in selected_queries:
            matched_fragments_content = []
            current_variable = query

            for _ in range(hops):
                if current_variable in variable_to_value:
                    next_value = variable_to_value[current_variable]
                    matched_fragments_content.append(
                        f"{current_variable} = {next_value}"
                    )
                    current_variable = next_value
                else:
                    break

            sample = MultiHopSample(
                query=query,
                matched_fragments_content=matched_fragments_content,
                all_fragments_content=all_fragments_content,
            )
            samples.append(sample)

        return samples

    @staticmethod
    def _make_levels(n: int, hops: int, string_length: int) -> List[dict]:
        levels = []
        # Generate Level 0
        level0 = {
            _make_random_string(length=string_length): _make_random_string(
                length=string_length
            )
            for _ in range(n)
        }
        level0_items = list(level0.items())
        random.shuffle(level0_items)
        levels.append(dict(level0_items))

        # Generate subsequent levels
        for _ in range(hops - 1):
            previous_values = list(levels[-1].values())
            new_level = {
                v: _make_random_string(length=string_length) for v in previous_values
            }
            new_level_items = list(new_level.items())
            random.shuffle(new_level_items)
            levels.append(dict(new_level_items))

        return levels


def _generate_dataset(
    num_samples: int,
    num_of_tokens: List[int],
    num_of_hops: List[int],
    hash_pair_str_length: List[int],
    output_path: str,
) -> None:
    random.seed(42)
    np.random.seed(42)

    chars_per_token = 3
    all_samples = []

    combinations = list(product(num_of_hops, hash_pair_str_length, num_of_tokens))
    total_combinations = len(combinations)

    base_samples_per_combination = num_samples // total_combinations
    remaining_samples = num_samples % total_combinations

    random.shuffle(combinations)

    sample_index = 0
    for idx, (hops, hash_length, tokens) in enumerate(
        tqdm(combinations, desc="Generating combinations")
    ):
        samples_count = base_samples_per_combination + (
            1 if idx < remaining_samples else 0
        )

        for _ in range(samples_count):
            samples = MultiHopEval.make_one(
                n_chars_problem=int(tokens * chars_per_token),
                num_queries=1,
                hops=hops,
                hash_pair_str_length=hash_length,
            )
            for sample in samples:
                # Build input text and expected output
                input_text = (
                    " ".join(sample.all_fragments_content)
                    + f"\nQuestion: What is the value of {sample.query}?"
                )
                # The expected output is the final value after following the chain
                if sample.matched_fragments_content:
                    final_value = sample.matched_fragments_content[-1].split(" = ")[-1]
                else:
                    final_value = "UNKNOWN"

                sample_dict = {
                    "index": sample_index,
                    "input": input_text,
                    "outputs": [final_value],
                    "length": len(input_text.split()),
                }
                json_line = json.dumps(sample_dict, ensure_ascii=False)
                all_samples.append(json_line)
                sample_index += 1

    # Verify that we have the correct number of samples
    assert (
        len(all_samples) == num_samples
    ), f"Expected {num_samples} samples, but got {len(all_samples)}"

    data_str = "\n".join(all_samples)
    write_file(data_str, output_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _app(config: DictConfig) -> None:
    print(f"Config:\n{OmegaConf.to_yaml(config)}")

    _generate_dataset(
        num_samples=config.num_samples,
        num_of_tokens=config.num_of_tokens,
        num_of_hops=config.num_of_hops,
        hash_pair_str_length=config.hash_pair_str_length,
        output_path=config.output_path,
    )
    print(f"Dataset generated and saved to {config.output_path}")


if __name__ == "__main__":
    _app()
