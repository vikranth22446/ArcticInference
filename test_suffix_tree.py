import os
import vllm
from vllm import LLM, SamplingParams
import pandas as pd
import json

# Fix CUDA device visibility issue
if os.environ.get('CUDA_VISIBLE_DEVICES') == 'all':
    # Get the number of available GPUs and set CUDA_VISIBLE_DEVICES accordingly
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))
        print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

vllm.plugins.load_general_plugins()
# Load data from parquet file
data_path = "/app/src/rllm/data/datasets/deepscaler_math/val_gsm8k_fixed_top20.parquet"
df = pd.read_parquet(data_path)

print(f"Loaded {len(df)} samples from {data_path}")
print("Dataset columns:", df.columns.tolist())

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 3,
        "enable_suffix_decoding": True,
        "disable_by_batch_size": 64,
    },
    trust_remote_code=True
)
def deep_search(obj, keywords, max_depth=8):
    visited = set()

    def _search(o, depth, path):
        if depth > max_depth:
            return
        oid = id(o)
        if oid in visited:
            return
        visited.add(oid)

        for attr in dir(o):
            if attr.startswith("__") and attr.endswith("__"):
                continue  # 跳过双下划线魔术方法
            try:
                value = getattr(o, attr)
            except Exception:
                continue

            # 模糊匹配：属性名中包含任意关键词就打印
            if any(k in attr.lower() for k in keywords):
                print(f"[FOUND] {attr} at path: {path}.{attr} (type={type(value)})")

            # 递归进入下一层
            _search(value, depth + 1, f"{path}.{attr}")

    _search(obj, 0, obj.__class__.__name__)


# 用法示例
keywords = ["model_executor", "driver_worker", "worker", "model_runner"]
deep_search(llm, keywords, max_depth=8)

def load_suffix_cache_data_for_problem_ids():
    suffix_cache_data_path = "/app/src/rllm/data/rollout_data/deepscaler_1.5b/hard20_16k__n8_1"
            
    problem_id_to_sequences = []
        # Check if the path exists
    if not os.path.exists(suffix_cache_data_path):
        print(f"Suffix cache data path does not exist: {suffix_cache_data_path}")
        return []
    # Handle both directory and file paths
    files_to_process = []
    if os.path.isdir(suffix_cache_data_path):
        # Search for JSONL files in the directory
        for filename in os.listdir(suffix_cache_data_path):
            if filename.endswith('.jsonl'):
                files_to_process.append(os.path.join(suffix_cache_data_path, filename))
    elif os.path.isfile(suffix_cache_data_path):
        files_to_process = [suffix_cache_data_path]
    
    if not files_to_process:
        print(f"No JSONL files found in suffix cache data path: {suffix_cache_data_path}")
        return []
    #print("DEBUG:Files found:", files_to_process)
    try:
        for file_path in files_to_process:
            print(f"Loading suffix cache data from: {file_path}")
            with open(file_path, 'r') as f:
                for line in f:
                    try:                            
                        data = json.loads(line.strip())
                        output_text = data['output']
                        problem_id_to_sequences.append(output_text)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON line in {file_path}: {line.strip()}, error: {e}")
                        continue
                        
    except Exception as e:
        print(f"Failed to load suffix cache data: {e}")
        
    print(f"Loaded suffix cache data for {len(problem_id_to_sequences)} problem IDs from {len(files_to_process)} files")
    return problem_id_to_sequences



# Load and bootstrap suffix cache data
#problem_id_to_sequences = load_suffix_cache_data_for_problem_ids()

try:
    # Try different ways to access the suffix cache based on vLLM version
    suffix_cache = None
    tokenizer = None
    print("DEBUG: attr of llm", dir(llm))
    print("DEBUG: attr of llm.llm_engine", dir(llm.llm_engine))
    print("DEBUG: attr of llm.llm_engine.engine_core", dir(llm.llm_engine.engine_core))
    
    # Check if engine_core has core_engine or core_engines
    if hasattr(llm.llm_engine.engine_core, 'core_engine'):
        print("DEBUG: attr of llm.llm_engine.engine_core.core_engine", dir(llm.llm_engine.engine_core.core_engine))
    if hasattr(llm.llm_engine.engine_core, 'core_engines'):
        print("DEBUG: attr of llm.llm_engine.engine_core.core_engines", dir(llm.llm_engine.engine_core.core_engines))
        if llm.llm_engine.engine_core.core_engines:
            print("DEBUG: attr of llm.llm_engine.engine_core.core_engines[0]", dir(llm.llm_engine.engine_core.core_engines[0]))
    
    # For vLLM v0.9.2: Check if llm_engine has model_executor directly
    print("DEBUG: Checking for model_executor in llm_engine...")
    if hasattr(llm.llm_engine, 'model_executor'):
        print("DEBUG: Found model_executor!")
        model_executor = llm.llm_engine.model_executor
        print("DEBUG: model_executor type:", type(model_executor))
        print("DEBUG: model_executor attributes:", [attr for attr in dir(model_executor) if not attr.startswith('_')])
        
        if hasattr(model_executor, 'driver_worker'):
            print("DEBUG: Found driver_worker!")
            driver_worker = model_executor.driver_worker
            print("DEBUG: driver_worker type:", type(driver_worker))
            print("DEBUG: driver_worker attributes:", [attr for attr in dir(driver_worker) if not attr.startswith('_')])
            
            if hasattr(driver_worker, 'model_runner'):
                print("DEBUG: Found model_runner!")
                model_runner = driver_worker.model_runner
                print("DEBUG: model_runner type:", type(model_runner))
                print("DEBUG: model_runner suffix-related attributes:", [attr for attr in dir(model_runner) if 'suffix' in attr.lower()])
                
                if hasattr(model_runner, '_suffix_cache'):
                    suffix_cache = model_runner._suffix_cache
                    print("DEBUG: Found suffix cache via model_executor.driver_worker.model_runner path!")
                else:
                    print("DEBUG: model_runner does not have _suffix_cache")
            else:
                print("DEBUG: driver_worker does not have model_runner")
        else:
            print("DEBUG: model_executor does not have driver_worker")
    else:
        print("DEBUG: llm_engine does not have model_executor")
    
    # Fallback: Check if we can access through engine_core directly
    if suffix_cache is None:
        print("DEBUG: Trying to access through engine_core...")
        if hasattr(llm.llm_engine, 'engine_core'):
            engine_core = llm.llm_engine.engine_core
            print(f"DEBUG: engine_core type: {type(engine_core)}")
            
            # Try to get the actual core engine object (not string)
            if hasattr(engine_core, 'core_engines') and engine_core.core_engines:
                print(f"DEBUG: core_engines length: {len(engine_core.core_engines)}")
                # Try to access the first core engine if it's not a string
                for i, core_eng in enumerate(engine_core.core_engines):
                    print(f"DEBUG: core_engines[{i}] type: {type(core_eng)}")
                    if not isinstance(core_eng, str):  # Skip if it's a string
                        print(f"DEBUG: Checking core_engines[{i}] for model_executor...")
                        if hasattr(core_eng, 'model_executor'):
                            print(f"DEBUG: Found model_executor in core_engines[{i}]!")
                            model_executor = core_eng.model_executor
                            if hasattr(model_executor, 'driver_worker'):
                                driver_worker = model_executor.driver_worker
                                if hasattr(driver_worker, 'model_runner'):
                                    model_runner = driver_worker.model_runner
                                    if hasattr(model_runner, '_suffix_cache'):
                                        suffix_cache = model_runner._suffix_cache
                                        print(f"DEBUG: Found suffix cache via core_engines[{i}] path!")
                                        break
    
    # Method 2: Try through engine_core attributes that might be engine objects
    if suffix_cache is None:
        print("DEBUG: Searching through all engine_core attributes...")
        for attr_name in dir(llm.llm_engine.engine_core):
            if not attr_name.startswith('_') and not callable(getattr(llm.llm_engine.engine_core, attr_name)):
                attr_obj = getattr(llm.llm_engine.engine_core, attr_name)
                if hasattr(attr_obj, 'model_executor'):
                    print(f"DEBUG: Found model_executor in engine_core.{attr_name}!")
                    model_executor = attr_obj.model_executor
                    if hasattr(model_executor, 'driver_worker'):
                        driver_worker = model_executor.driver_worker
                        if hasattr(driver_worker, 'model_runner'):
                            model_runner = driver_worker.model_runner
                            if hasattr(model_runner, '_suffix_cache'):
                                suffix_cache = model_runner._suffix_cache
                                print(f"DEBUG: Found suffix cache via engine_core.{attr_name} path!")
                                break
    
    # Method 3: Try to find any object with model_executor in the entire engine structure
    if suffix_cache is None:
        print("DEBUG: Deep search for any object with model_executor...")
        def deep_search_for_model_executor(obj, path="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return None
            
            if hasattr(obj, 'model_executor'):
                model_executor = obj.model_executor
                if hasattr(model_executor, 'driver_worker'):
                    driver_worker = model_executor.driver_worker  
                    if hasattr(driver_worker, 'model_runner'):
                        model_runner = driver_worker.model_runner
                        if hasattr(model_runner, '_suffix_cache'):
                            print(f"DEBUG: Found suffix cache via {path}.model_executor path!")
                            return model_runner._suffix_cache
            
            # Search through attributes
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr_obj = getattr(obj, attr_name)
                        if not callable(attr_obj) and attr_obj is not None:
                            if isinstance(attr_obj, list) and len(attr_obj) > 0:
                                for i, item in enumerate(attr_obj[:3]):  # Check first 3 items
                                    result = deep_search_for_model_executor(item, f"{path}.{attr_name}[{i}]", max_depth, current_depth + 1)
                                    if result is not None:
                                        return result
                            else:
                                result = deep_search_for_model_executor(attr_obj, f"{path}.{attr_name}", max_depth, current_depth + 1)
                                if result is not None:
                                    return result
                    except:
                        continue
            return None
        
        suffix_cache = deep_search_for_model_executor(llm.llm_engine, "llm.llm_engine")
    
    # Get tokenizer
    if hasattr(llm.llm_engine, 'tokenizer'):
        tokenizer = llm.llm_engine.tokenizer
    elif hasattr(llm, 'tokenizer'):
        tokenizer = llm.tokenizer
    
    # Method 5: Recursive search for _suffix_cache
    if suffix_cache is None:
        def find_suffix_cache(obj, path="", max_depth=5, current_depth=0):
            if current_depth >= max_depth:
                return None
            
            if hasattr(obj, '_suffix_cache'):
                print(f"DEBUG: Found _suffix_cache at path: {path}._suffix_cache")
                return getattr(obj, '_suffix_cache')
            
            # Search through common attributes
            for attr_name in ['model_executor', 'driver_worker', 'model_runner', 'core_engine', 'core_engines', 'workers']:
                if hasattr(obj, attr_name):
                    attr_obj = getattr(obj, attr_name)
                    if attr_obj is not None:
                        if isinstance(attr_obj, list) and len(attr_obj) > 0:
                            # If it's a list, check the first element
                            result = find_suffix_cache(attr_obj[0], f"{path}.{attr_name}[0]", max_depth, current_depth + 1)
                            if result is not None:
                                return result
                        else:
                            result = find_suffix_cache(attr_obj, f"{path}.{attr_name}", max_depth, current_depth + 1)
                            if result is not None:
                                return result
            return None
        
        print("DEBUG: Starting recursive search for _suffix_cache...")
        suffix_cache = find_suffix_cache(llm.llm_engine, "llm.llm_engine")
    
    if suffix_cache is not None and tokenizer is not None:
        print(f"DEBUG: Successfully found suffix cache and tokenizer, bootstrapping with {len(problem_id_to_sequences)} sequences")
        for output_index, output_text in enumerate(problem_id_to_sequences):
            output_index += 1
            # Tokenize the output text to get token sequences
            token_ids = tokenizer.encode(output_text, add_special_tokens=False)
            if token_ids:  # Only update if we have valid tokens
                suffix_cache.update_response(req_id=-output_index-1, token_ids=token_ids)
                print(f"DEBUG: Bootstrap updated suffix cache with {len(token_ids)} tokens, first_10_tokens={token_ids[:10]}")
    else:
        print("WARNING: Could not find suffix cache or tokenizer - suffix cache bootstrapping skipped")
        print(f"DEBUG: suffix_cache found: {suffix_cache is not None}")
        print(f"DEBUG: tokenizer found: {tokenizer is not None}")
        
        # If we still can't find it, let's see what attributes are available in the deepest objects
        if suffix_cache is None:
            print("DEBUG: Exploring deeper structure...")
            if hasattr(llm.llm_engine.engine_core, 'core_engines') and llm.llm_engine.engine_core.core_engines:
                core_engine = llm.llm_engine.engine_core.core_engines[0]
                if hasattr(core_engine, 'model_executor'):
                    print("DEBUG: core_engine.model_executor attributes:", dir(core_engine.model_executor))
                    if hasattr(core_engine.model_executor, 'driver_worker'):
                        print("DEBUG: core_engine.model_executor.driver_worker attributes:", dir(core_engine.model_executor.driver_worker))
                        if hasattr(core_engine.model_executor.driver_worker, 'model_runner'):
                            print("DEBUG: core_engine.model_executor.driver_worker.model_runner attributes:", dir(core_engine.model_executor.driver_worker.model_runner))
        
except Exception as e:
    print(f"Failed to bootstrap suffix cache: {e}")
    import traceback
    traceback.print_exc()

# Set n_sampling = 8
n_sampling = 8
sampling_params = SamplingParams(temperature=0.6, max_tokens=16000, n=n_sampling)

# Process each sample in the dataset
for idx, row in df.iterrows():
    print(f"\n=== Processing sample {idx + 1}/{len(df)} ===")
    
    # Assuming the dataset has a 'question' or 'prompt' column
    # You may need to adjust the column name based on the actual structure
    if 'question' in df.columns:
        prompt = row['question']
    elif 'prompt' in df.columns:
        prompt = row['prompt']
    elif 'input' in df.columns:
        prompt = row['input']
    else:
        # Use the first text column if standard names don't exist
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            prompt = row[text_columns[0]]
        else:
            print(f"Could not find text column in row {idx}")
            continue
    
    conversation = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    print(f"Input: {prompt}")
    
    # Generate n_sampling responses
    #outputs = llm.chat(conversation, sampling_params=sampling_params)
    
    # print(f"Generated {len(outputs[0].outputs)} responses:")
    # for i, output in enumerate(outputs[0].outputs):
    #     print(f"Response {i+1}: {output.text}")
    #     print("-" * 50)
    
    # Break after first sample for testing - remove this line to process all samples
    break
    