import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 10  # seconds.
app = FastAPI()
engine = None


def tactic_prompt_1(tactic_state, prefix, context):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Here are some examples:

Tactic state:
---
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
---
Next tactic:
---
rintro s t ⟨u, a, hr, he⟩
---

Tactic state:
---
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
---
Next tactic:
---
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
---

Tactic state:
---
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
---
Next tactic:
---
rw [← h.gcd_eq_one]
---

Tactic state:
---
%s
---
Next tactic:
---
%s""" % (tactic_state, prefix)
    return prompt


def tactic_prompt_2(tactic_state, prefix, context):
    prompt = context + prefix
    return prompt

PROMPTS = {
    'default' : {
        'tactic' : [tactic_prompt_1, tactic_prompt_2]
    }
}

def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _filter(texts, scores):
    texts_ = []
    scores_ = []
    for text, score in zip(texts, scores):
        if text.strip() in {"", "sorry", "admit"}:
            continue
        texts_.append(text)
        scores_.append(score)
    return texts_, scores_

def check_key(key):
    return True


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/tactic")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - model: the model name.
    - key: the authorization key.
    - tactic_state: the tactic state.
    - context: context string.
    - prefix: prefix string.
    - other parameters: sampling parameters
    """
    request_dict = await request.json()

    model = request_dict.pop('model', 'default')
    tactic_state = request_dict.pop('tactic_state', '')
    context = request_dict.pop('context', '')
    prefix = request_dict.pop('prefix', '')
    key = request_dict.pop('key', '')

    if not check_key(key):
       return JSONResponse({"outputs": []}) 

    sampling_params = SamplingParams(**request_dict)
    texts, scores = [], []
    for prompt_fn in PROMPTS.get(model, 'default')['tactic']:
        request_id = random_uuid()
        prompt = prompt_fn(tactic_state, prefix, context)
        results_generator = engine.generate(prompt, sampling_params, request_id)

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output

        assert final_output is not None
        for output in final_output.outputs:
            text = output.text #.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)
        texts, scores = _unique_sorted(texts, scores)
        texts, scores = _filter(texts, scores)

    outputs = list(zip(texts, scores))
    ret = {"outputs": outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)