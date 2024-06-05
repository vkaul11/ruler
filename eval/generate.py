import json
import aiohttp
import asyncio
import hydra
from omegaconf import DictConfig
import os
from hydra.utils import get_original_cwd
import aiofiles  # Import aiofiles for async file operations

async def chat_completion(config: DictConfig, messages: list[dict[str, str]], session: aiohttp.ClientSession) -> str:
    payload = {
        "model": config.model_id,
        **config.prediction_params,
        "messages": messages,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": config.auth_key
    }
    url = config.url
    async with session.post(url, json=payload, headers=headers) as response:
        result = await response.json()
        return result['choices'][0]['message']['content']
    
def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def get_metric_function(metric_type):
    if metric_type == 'partial':
        return string_match_part
    else:
        return string_match_all


async def _process_data(input_data, config, output_file, error_file):
    metrics = []
    metric_function = get_metric_function(config.task.metric_type)
    async with aiohttp.ClientSession() as session, aiofiles.open(output_file, 'w') as file, aiofiles.open(error_file, 'w') as error_file:
        semaphore = asyncio.Semaphore(6)  # Control concurrency
        tasks = [handle_request(config, data, semaphore, session, file, metrics, error_file, metric_function) for data in input_data]
        await asyncio.gather(*tasks)
        # Compute average score after all tasks are completed
        if metrics:
            average_score = sum(metrics) / len(metrics)
            print("Average Metric Score:", average_score)

async def handle_request(config, data, semaphore, session, file, metrics, error_file, metrics_function):
    async with semaphore:
        messages = [{"role": "user", "content": data["input"]}]
        prediction = await chat_completion(config, messages, session)
        metric_score = metrics_function([prediction], [data["outputs"]])
        metrics.append(metric_score)  # Collect score for averaging later
        output_data = {
            "index": data["index"],
            "input": data["input"],
            "outputs": data["outputs"],
            "pred": prediction,
            "score": metric_score
        }
        await file.write(json.dumps(output_data) + '\n')
        if metric_score < 100.0:
            await error_file.write(json.dumps(output_data) + '\n')  # Write error results to another file
        print(f"Processed index {data['index']} with score: {metric_score}")

@hydra.main(config_path='conf', config_name='config', version_base=None)
def _app(config: DictConfig) -> None:
    input_data = []
    original_cwd = get_original_cwd()
    input_path = os.path.abspath(os.path.join(original_cwd, config.task.input_file_path))
    output_path = os.path.join(original_cwd, config.task.output_file)
    errors_path = os.path.join(original_cwd, config.task.error_file)  

    with open(input_path, 'r') as file:
        for line in file:
            input_data.append(json.loads(line))

    # Asyncio event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_process_data(input_data, config, output_path, errors_path))

if __name__ == "__main__":
    _app()
