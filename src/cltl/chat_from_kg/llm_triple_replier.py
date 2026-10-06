import json
import os

from prompts.response_processor import PromptProcessor


def _load_openai_key() -> str:
    env = os.environ.get("OPENAI_API_KEY")
    if not env:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")
    return env


DEFAULT_TIMEOUT = 60.0  # seconds, backend="openai" only -- see the `timeout` parameter docstring below


class LLMTripleReplier():
    def __init__(self, language="English", model_name="llama3.2", backend="ollama", openai_model="gpt-5.1",
                 timeout: float = DEFAULT_TIMEOUT):
        """
        Generate natural language based on structured data

        Parameters
        ----------
        language: language to reply in.
        model_name: Ollama model name, used only when backend="ollama".
        backend: "ollama" (default, needs a local Ollama server -- langchain_ollama is only
            imported in that case) or "openai" (needs OPENAI_API_KEY, same as the rest of
            events_from_chat).
        openai_model: OpenAI chat model, used only when backend="openai".
        timeout: per-request timeout (seconds), backend="openai" only -- passed straight
            through to openai.OpenAI(timeout=...). Without this, a stuck or very slow request
            has no ceiling and can block reply() indefinitely (this is what happened before it
            was added -- a live chat turn once hung for 5+ minutes with no error). max_retries=0
            too, so a timeout surfaces immediately as openai.APITimeoutError instead of the SDK
            silently retrying (by default, up to 2 more times) before finally raising --
            chat_sessions.KgChatSession's _call_openai() catches that and turns it into a
            message the human sees. Not used for backend="ollama" (ChatOllama has its own,
            unrelated timeout handling).
        """
        self._language = language
        self._processor = PromptProcessor(language)
        self._backend = backend

        if backend == "ollama":
            # Imported lazily so backend="openai" works without langchain_ollama installed or
            # a local Ollama server running.
            from langchain_ollama import ChatOllama
            self._ollama_client = ChatOllama(model=model_name, temperature=0.1, num_ctx=4096)  # limits KV-cache size, avoids GPU OOM
        elif backend == "openai":
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=_load_openai_key(), timeout=timeout, max_retries=0)
            self._openai_model = openai_model
        else:
            raise ValueError(f"Unknown backend {backend!r}, expected 'ollama' or 'openai'")

    def reply(self, prompt) -> str:
        """Send one [instruct, content] prompt -- as built by PromptProcessor's get_*_prompt
        methods, e.g. get_all_prompt_input_from_response() or get_prompt_for_kg_gap() -- to
        this replier's LLM backend and return the plain-text reply."""
        if self._backend == "ollama":
            return self._ollama_client.invoke(prompt).content
        response = self._openai_client.chat.completions.create(model=self._openai_model, messages=prompt)
        return response.choices[0].message.content


if __name__ == "__main__":
    model_name = "llama3.2"
    replier = LLMTripleReplier(language="Dutch", model_name=model_name)
    path = "../../../data/thoughts-responses.json"
    print(path)
    file = open(path)
    data = json.load(file)
    for response in data:
        prompts = replier._processor.get_all_prompt_input_from_response(response)
        for prompt in prompts:
            print("PROMPT", prompt)
            print('RESPONSE:', replier.reply(prompt))
