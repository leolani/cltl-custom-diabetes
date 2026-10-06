# cltl-custom-diabetes

A custom module for the cltl-apps platform that runs the
[cltl-kg-driven-chat](../../cltl-kg-driven-chat) diabetes lifestyle coach as
one tenant's conversational agent. Instead of starting cold, the agent opens
every conversation by:

1. looking up in the knowledge graph **when it last spoke with the human**,
2. summarising **the topics discussed back then**, and
3. asking about the topic that most needs catching up on.

After that, every utterance is extracted into events, pushed to the knowledge
graph, and followed up with the questions defined in `intents/*.json`. Once an
activity's intent has no more open requirements, the agent goes back to the
next catch-up topic until every topic has had enough coverage for the period.

When the human asks about their own record ("what did I eat yesterday?", "when
did I last go for a walk?"), the agent answers from the knowledge graph instead
(`src/kg_chat/kg_question_answerer.py`). An LLM turns the question into a query
plan: activity types, keywords and a start date. Fixed SPARQL queries fetch the
activities the human reported. The LLM then paraphrases them as a reply, and a
reply over `max_answer_words` is summarised. The question and its answer are
not pushed to the graph.

The UI of cltl-kg-driven-chat is not used; the platform's chat UI is.

## How it attaches to the platform

Like `cltl-custom-leoric`, this module only subscribes and publishes on the
event bus. Nothing in the platform setup changes.

| Topic | Direction | Used for |
|---|---|---|
| `cltl.topic.text_in` | in | the human's utterances (typed or ASR) |
| `cltl.topic.scenario` | in | `ScenarioStarted` / `ScenarioStopped` |
| `cltl.topic.text_out` | out | the agent's replies |

Each scenario gets its own conversation. With the default `open_on: utterance`,
the first utterance in a scenario (e.g. the "yes" to the platform's greeting)
opens the conversation with the catch-up message. That first utterance is not
answered separately. With `open_on: scenario`, the catch-up message is sent as
soon as the scenario starts. When the scenario stops, the turns and the intent
log are written to `logs/{chat_logs,intents_log}`.

Tenancy works as for any other client stack: the module binds the routing keys
of `CLTL_TENANT` and copies the tenant of each incoming event onto its replies.

**Run it as its own tenant.** In a tenant where `cltl-custom-leoric` also runs,
both modules would answer every `text_in`.

## Requirements

- A running cltl-apps deployment (`cltl-apps-restructure_monitoring`): the
  broker and servers, plus the tenant's `clients/*` stacks (backend, context,
  chat-ui).
- A GraphDB repository with the conversation history, by default
  `http://localhost:7200/repositories/event_sandbox` (the one the
  kg-driven-chat notebooks use). On first use, the module uploads the
  `n2mu_sem_roles` role hierarchy if it is missing.
- `OPENAI_API_KEY`, used for event extraction and reply generation.

## Configuration

`[cltl.diabetes]` in `config/default.config`:

| Key | Default | Meaning |
|---|---|---|
| `human` | `Jan` | The person the agent talks to, as named in the knowledge graph |
| `use_scenario_speaker` | `False` | Use the speaker name from `ScenarioStarted` instead, when it carries one |
| `open_on` | `utterance` | `utterance` or `scenario`, see above |
| `kg_server` | `http://localhost:7200` | GraphDB server |
| `kg_repository` | `event_sandbox` | GraphDB repository; the endpoint is `<kg_server>/repositories/<kg_repository>` |
| `kg_address` | `$CLTL_KG_ADDRESS` | Full SPARQL endpoint; when set, it takes precedence over `kg_server`/`kg_repository` |
| `intents_dir` | `./intents` | Intent definitions driving the follow-up questions |
| `log_dir` | `./logs` | Where chat, intent and brain logs go |
| `fallback_gap_days` | `7` | Assumed gap when there is no earlier conversation in the graph |
| `model` | (empty) | OpenAI model; empty uses kg-driven-chat's default |
| `answer_questions` | `True` | Answer the human's questions about their own record from the knowledge graph |
| `max_answer_words` | `80` | Answers longer than this are summarised |

Override any of these in `config/custom.config`.

## Running

Bring up the platform and the tenant's clients as described in
`../cltl-apps-restructure_monitoring/README.md`. Then start this module for the
same tenant (`CLTL_TENANT` from `config/clients.env`), in one of two ways.

### As a local process

Python 3.10 (emissor needs it):

```bash
cd cltl-custom-diabetes
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

OPENAI_API_KEY=... \
CLTL_AMQP_URL=amqp://eliza:eliza123@127.0.0.1:5672/ CLTL_TENANT=tenant-a \
    python src/main.py
```

The repository comes from `kg_server`/`kg_repository` in `config/custom.config`.
Setting `CLTL_KG_ADDRESS` to a full SPARQL endpoint overrides them.

Run it from this directory. The kg-driven-chat code finds `src/cltl` and
`intents/` relative to the working directory.

### As a container on the platform network

```bash
cd cltl-custom-diabetes
export OPENAI_API_KEY=...
docker compose --env-file ../cltl-apps-restructure_monitoring/config/clients.env \
    -f compose/diabetes.compose.yml up -d --build
```

The container reaches the broker as `rabbitmq` on the `cltl-platform` network,
and GraphDB on the host as `host.docker.internal:7200`.

Then open the tenant's chat UI (e.g. <http://localhost:8003/chatui/static/chat.html>)
and say something to start the conversation.

## Running without the platform

The original kg-driven-chat application runs from the notebooks, with its own
Tkinter chat window instead of the platform's chat UI. It needs no broker,
tenant or `config/`, only GraphDB and `OPENAI_API_KEY`:

```bash
cd cltl-custom-diabetes
pip install -r requirements.txt
pip install jupyter pandas    # or: pip install -e ".[notebooks]"
cd notebooks
OPENAI_API_KEY=... jupyter notebook
```

Start Jupyter from `notebooks/`. Each notebook's first code cell puts
`src/kg_chat` and `src/` on `sys.path`, and the kg-driven-chat code finds
`src/cltl` and `intents/` by walking up from the working directory. The
GraphDB repository is the `KG_ADDRESS` set in each notebook.

| Notebook | What it runs |
|---|---|
| `chat_session.ipynb` | A plain LLM chat, no knowledge graph |
| `kg_chat_session.ipynb` | Events pushed to the KG, follow-up questions from gaps in the KG |
| `kg_intent_chat.ipynb` | Follow-up questions from `intents/*.json` |
| `kg_catchup_intent_chat.ipynb` | The catch-up flow that this module runs on the platform |

The notebooks write their logs to `notebooks/{chat_logs,intents_log,kg_logs}`.
`doc/` has the slides that explain the agent (PDF, HTML and PowerPoint).

## Layout

```
src/cltl/diabetes/    this module: the bus service (service.py), its wiring
                      (container.py), and the bridge to kg-driven-chat (catchup.py)
src/kg_chat/          chat_sessions.py, catch_up_from_kg.py  ┐
src/cltl/             chat_from_kg, events_from_chat,         │ copied
                      gaps_from_kg                            │ from
intents/              *_intents.json                          │ cltl-kg-driven-chat
notebooks/            the notebooks and kg_chat_gui.py        │
doc/                  slides on the KG-driven chat agent      ┘
src/kg_chat/          kg_question_answerer.py: answers questions from the KG
src/cltl/commons/     copied from cltl-combot main, see below
config/               default.config (module settings), custom.config(.example)
                      (broker and tenant), logging.config
compose/              diabetes.compose.yml
tests/                service tests against a fake conversation
```

`src/cltl/commons` is copied from cltl-combot's `main` branch. The platform
needs cltl-combot's `ref_scenarios` branch for tenants and scenario ids, but
that branch's `cltl.commons.discrete` lacks `Level` and `Polarity.EXPECT`,
which `cltl.brain` and the kg-chat code import. Nothing in cltl-combot itself
imports `cltl.commons`, and `src/` comes first on the path, so this copy
replaces the one from cltl-combot. Remove it once both branches agree.

The kg-driven-chat code is copied unchanged, with these exceptions:

- `src/kg_chat/chat_sessions.py` has an optional `question_answerer` that
  answers the human's questions about their own record from the knowledge
  graph.
- Each notebook has an extra first code cell that puts `src/kg_chat` and
  `src/` on `sys.path`, because `chat_sessions.py` and `catch_up_from_kg.py`
  no longer sit next to the notebooks.
- `kg_intent_chat.ipynb` and `kg_catchup_intent_chat.ipynb` pass a
  `KgQuestionAnswerer` to the session, so a question such as "what did I
  drink?" is answered from the knowledge graph. Without it, the question is
  extracted as a new drink activity and answered with a follow-up question.
  `notebooks/kg_chat_gui.py` labels those answers `[KG answer]`.

The notebooks' run logs (`chat_logs/`, `intents_log/`, `kg_logs/`) were not
copied.

To pick up changes from that repository, copy it again. Keep `diabetes` and
`commons` out of the `--delete`, and merge `chat_sessions.py` by hand instead of
overwriting it:

```bash
K=../../cltl-kg-driven-chat
rsync -a --delete --exclude __pycache__ --exclude .DS_Store \
    --exclude /diabetes/ --exclude /commons/ $K/src/cltl/ src/cltl/
cp $K/notebooks/catch_up_from_kg.py src/kg_chat/
cp $K/notebooks/*.ipynb $K/notebooks/kg_chat_gui.py $K/notebooks/diabetes2Jan.png notebooks/
diff $K/notebooks/chat_sessions.py src/kg_chat/chat_sessions.py
rsync -a --delete --exclude .ipynb_checkpoints $K/intents/ intents/
cp $K/doc/*.pdf $K/doc/*.html $K/doc/*.pptx doc/
```

Copying the notebooks again overwrites these changes, so add them back.

## Tests

```bash
PYTHONPATH=src pytest tests
```

## License

MIT, see [`LICENSE`](LICENSE).
