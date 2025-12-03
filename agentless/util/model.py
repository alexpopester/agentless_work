import json
from abc import ABC, abstractmethod
from typing import List
import os
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from agentless.util.api_requests import (
    create_anthropic_config,
    create_chatgpt_config,
    request_anthropic_engine,
    request_chatgpt_engine,
)

# Check for llama-cpp-python availability
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

MADE_MODEL = None


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    _STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    _USER_REPLY_EDIT_MESSAGE = """File is successfully edited"""

    tools = [
        {
            "name": "str_replace_editor",
            "description": _STR_REPLACE_EDITOR_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
            },
        }
    ]

    MAX_CODEGEN_ITERATIONS = 10

    # specialized codegen with tool
    def codegen_w_tool(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        def _build_response_and_extract(response, messages, iter):
            json_response = response.to_dict()

            contains_tool = False
            # formulate the messages
            json_response.pop("id")
            json_response.pop("model")
            json_response.pop("stop_reason")
            json_response.pop("stop_sequence")
            json_response.pop("type")
            json_response.pop("usage")

            messages.append(json_response)

            response_content = []

            for json_message in json_response["content"]:
                if json_message["type"] == "tool_use":
                    contains_tool = True
                    # each tool use requires a response
                    response_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": json_message["id"],
                            "content": self._USER_REPLY_EDIT_MESSAGE,
                        }
                    )

            if contains_tool:
                messages.append(
                    {
                        "role": "user",
                        "content": response_content,
                    }
                )
            else:
                if iter == 0:
                    # if the first iteration does not contain the tool, likely the model is doing some CoT for debugging
                    # append encouraging message
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please generate editing commands to fix the issue",
                                }
                            ],
                        }
                    )
                    contains_tool = True

            return messages, contains_tool

        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            print(f" === Generating ====")
            # initialized the traj
            traj = {
                "response": [],
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "cache_creation_token": 0,
                    "cache_read_input_tokens": 0,
                },
            }

            # create the initial config and messages
            messages = [
                {"role": "user", "content": [{"type": "text", "text": message}]}
            ]

            for iteration in range(self.MAX_CODEGEN_ITERATIONS):
                config = create_anthropic_config(
                    message=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=1,
                    model=self.name,
                    tools=self.tools,
                )
                ret = request_anthropic_engine(
                    config,
                    self.logger,
                    prompt_cache=True,  # prompt cache should be always true as we at least should query twice
                )

                if ret:
                    # add the response to the traj
                    traj["response"].append([reply.to_dict() for reply in ret.content])

                    # pretty dump the response
                    for reply in ret.content:
                        print(json.dumps(reply.to_dict(), indent=2))

                    # update the usage
                    traj["usage"]["completion_tokens"] += ret.usage.output_tokens
                    traj["usage"]["prompt_tokens"] += ret.usage.input_tokens
                    traj["usage"][
                        "cache_creation_token"
                    ] += ret.usage.cache_creation_input_tokens
                    traj["usage"][
                        "cache_read_input_tokens"
                    ] += ret.usage.cache_read_input_tokens

                    messages, contains_tool = _build_response_and_extract(
                        ret, messages, iteration
                    )

                    if not contains_tool:
                        break
                else:
                    assert (
                        False
                    ), "No response from the engine"  # this should not happen

            if ret:
                trajs.append(traj)
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_anthropic_engine(
                config, self.logger, prompt_cache=prompt_cache
            )

            if ret:
                trajs.append(
                    {
                        "response": ret.content[0].text,
                        "usage": {
                            "completion_tokens": ret.usage.output_tokens,
                            "prompt_tokens": ret.usage.input_tokens,
                            "cache_creation_token": (
                                0
                                if not prompt_cache
                                else ret.usage.cache_creation_input_tokens
                            ),
                            "cache_read_input_tokens": (
                                0
                                if not prompt_cache
                                else ret.usage.cache_read_input_tokens
                            ),
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class GeminiChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

        # Configure the API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.logger.warning("GOOGLE_API_KEY is not set in environment variables.")
        genai.configure(api_key=api_key)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        # Configure the model (e.g., "gemini-1.5-pro-latest")
        model = genai.GenerativeModel(self.name)

        # Set generation config
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,  # Gemini usually generates 1 per request via SDK
            max_output_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        # CRITICAL: Disable safety settings for code generation.
        # Code (like 'rm -rf' or hacky scripts) often triggers false positives in default filters.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        trajs = []
        print(f"Getting {num_samples} samples")
        for _ in range(num_samples):
            try:
                # Agentless usually passes the whole context as 'message'.
                # We treat it as a chat message or a raw prompt.
                response = model.generate_content(
                    contents=message,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                # print(f"generated response: {response}")

                # Extract usage metadata if available
                usage = {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                }

                if response.usage_metadata:
                    usage["completion_tokens"] = (
                        response.usage_metadata.candidates_token_count
                    )
                    usage["prompt_tokens"] = response.usage_metadata.prompt_token_count

                # Check if response was blocked or empty
                if response.text:
                    trajs.append({"response": response.text, "usage": usage})
                else:
                    self.logger.warning(
                        f"Gemini response was empty or blocked: {response.prompt_feedback}"
                    )
                    print("BLOCKED")
                    trajs.append({"response": "", "usage": usage})

            except Exception as e:
                self.logger.error(f"Error calling Gemini API: {e}")
                print(f"Error calling Gemini API: {e}")
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )
                # Basic rate limit backoff if needed
                time.sleep(1)

        return trajs

    def is_direct_completion(self) -> bool:
        return False


import requests
import json
import os


class LocalChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)
        # Default to standard llama-server port
        self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:

        # 1. Validate inputs
        if self.temperature == 0:
            num_samples = 1

        # 2. Construct the Endpoint URL
        # We ensure no double slashes if base_url ends in /
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"

        # 3. Manually build the payload (mimics OpenAI API)
        payload = {
            "model": self.name,  # The server usually ignores this or treats it as an alias
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "n": num_samples,  # Ask the server to generate 'n' choices
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # 4. Send the raw HTTP Request
            print("Posting")
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=300,  # Generous timeout for local inference
            )
            print("Got post")
            response.raise_for_status()  # Raise error for 400/500 codes

            print("Pulled Data")
            data = response.json()

            # 5. Extract Data
            choices = data.get("choices", [])
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

        except Exception as e:
            self.logger.error(f"Local LLM Request failed: {e}")
            print(f"Request failed: {e}")
            # Return empty structure on failure to prevent pipeline crash
            return [
                {"response": "", "usage": {"completion_tokens": 0, "prompt_tokens": 0}}
            ] * num_samples

        # 6. Format the output to match DecoderBase expectation
        trajs = []

        # We assume the API returned 'num_samples' choices.
        # If it returned fewer, we loop over what we got.
        print(f"Enumerating choices : {len(choices)}")
        for i, choice in enumerate(choices):
            text = choice.get("message", {}).get("content", "")

            # Logic from your previous snippet:
            # Attribute usage costs only to the first sample to avoid double counting
            c_tokens = completion_tokens if i == 0 else 0
            p_tokens = prompt_tokens if i == 0 else 0

            trajs.append(
                {
                    "response": text,
                    "usage": {
                        "completion_tokens": c_tokens,
                        "prompt_tokens": p_tokens,
                    },
                }
            )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class LlamaCppDecoder(DecoderBase):
    """
    Runs the model using llama-cpp-python (GGUF format).
    The 'name' parameter should be the path to the .gguf file.
    """

    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. Please run `pip install llama-cpp-python`."
            )

        if not os.path.exists(name):
            self.logger.warning(
                f"GGUF model path not found locally: {name}. Ensure this is a valid path to a .gguf file."
            )

        print(f"Loading GGUF model from {name}...")

        # Initialize Llama
        # n_gpu_layers=-1 attempts to offload all layers to GPU
        # n_ctx=4096 sets context window (default is often small like 512)
        self.llm = Llama(
            model_path=name,
            n_gpu_layers=-1,
            n_ctx=8192,  # Set higher for coding tasks
            verbose=False,
        )

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:

        trajs = []
        # llama-cpp-python usually generates one at a time unless using batched inference which is complex
        # We loop for num_samples
        for _ in range(num_samples):
            # Create completion
            output = self.llm(
                message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop=[],  # Add stop tokens if necessary, e.g. ["<|im_end|>"]
            )

            # Extract standard OpenAI-like usage stats
            usage = output.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            text = output["choices"][0]["text"]

            trajs.append(
                {
                    "response": text,
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                    },
                }
            )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    global MADE_MODEL
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "gemini":
        return GeminiChatDecoder(
            name="gemini-2.5-flash-lite",
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "local":
        return LocalChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "llama_cpp":
        if MADE_MODEL != None:
            return MADE_MODEL
        returned_model = LlamaCppDecoder(
            name=os.environ["PATH_TO_LOCAL_MODEL"],
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        if MADE_MODEL == None:
            MADE_MODEL = returned_model
        return returned_model
    else:
        raise NotImplementedError
