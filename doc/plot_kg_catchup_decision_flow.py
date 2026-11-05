"""Draw the dialogue decision process of notebooks/kg_catchup_intent_chat.ipynb as a Graphviz flowchart.

Follows NameFirstCatchUpSession (src/cltl/kg_chat/name_first_session.py; in src/main.py:
AskNameConversation/CatchUpConversation, src/cltl/diabetes/catchup.py), KgIntentChatSession.say()
(src/cltl/kg_chat/chat_sessions.py) and wrap_agent_fn_with_saturation_loop() / SaturationTracker
(src/cltl/kg_chat/catch_up_from_kg.py). Every LLM call goes to OpenAI or a local llama-server
(llm_backend; `llm_backend` in config/default.config, LLM_BACKEND in the notebook).

Needs the Graphviz binaries (`brew install graphviz`) and the `graphviz` Python package.

    .venv/bin/python doc/plot_kg_catchup_decision_flow.py
    ->  doc/kg_catchup_decision_flow.png  and  doc/kg_catchup_decision_flow.svg
"""
from pathlib import Path

import graphviz

OUT = Path(__file__).with_name("kg_catchup_decision_flow")

C_SETUP = "#dbe4f0"
C_DEC = "#fff2c2"
C_ACT = "#d8efd9"
C_CATCH = "#fbdcc4"
C_IO = "#e6e0f3"


def tagged(text, tag):
    """Node label with a small monospace reply_sources tag underneath."""
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = "<BR/>".join(esc.split("\n"))
    return f'<{lines}<BR/><FONT FACE="Courier" POINT-SIZE="9" COLOR="#666666">[{tag}]</FONT>>'


g = graphviz.Digraph("kg_catchup_decision_flow")
g.attr(rankdir="TB", newrank="true", nodesep="0.5", ranksep="0.45", fontname="Helvetica",
       label="KG catch-up + intent chat: dialogue decision process\n ", labelloc="t", fontsize="20",
       bgcolor="white", pad="0.3")
g.attr("node", fontname="Helvetica", fontsize="11", style="filled,rounded", shape="box",
       color="#444444", penwidth="1.2", margin="0.15,0.08")
g.attr("edge", fontname="Helvetica-Bold", fontsize="10", color="#444444", arrowsize="0.8")


def decision(sg, name, text):
    sg.node(name, text, shape="diamond", style="filled", fillcolor=C_DEC, margin="0.02", group="spine")


# ---------------- Before the chat ----------------
with g.subgraph(name="cluster_setup") as c:
    c.attr(label="Before the catch-up: who is it, and what does the KG remember?", style="dashed,filled",
           fillcolor="#f3f6fa", color="#b8c4d6", fontname="Helvetica-Oblique", fontsize="11", labeljust="l")
    c.node("known", "Opening: name everyone the agent\nhas talked with (KG) and ask\n\"What's your name?\"",
           fillcolor=C_CATCH)
    c.node("name", "Name from the answer:\na known name (any case), else\nthe LLM picks it out; none: ask again",
           fillcolor=C_SETUP)
    decision(c, "d_known", "Known\nperson?")
    c.node("memory", "Memory (KG queries for this person):\nlast conversation; catch-up topics +\n"
           "targets (period capped at 14 days);\ntopics of the last chat (own words);\n"
           "when each topic was last discussed\n-> timeline shown in the GUI", fillcolor=C_SETUP)
    c.node("open", "Catch-up opening:\ntime since last chat, the topics\nof the LAST chat only, question\n"
           "on the most under-covered topic", fillcolor=C_CATCH)
    c.node("welcome", "Welcome (first meeting):\nno catch-up topics, no gap;\nask how they have been",
           fillcolor=C_CATCH)
    with c.subgraph() as r:
        r.attr(rank="same")
        for n in ("known", "name", "d_known", "memory", "open"):
            r.node(n)
    c.edge("known", "name")
    c.edge("name", "d_known")
    c.edge("d_known", "memory", label=" yes")
    c.edge("memory", "open")
    c.edge("d_known", "welcome", label=" no")

g.node("human", "Human utterance", fillcolor=C_IO, group="spine", fontname="Helvetica-Bold", fontsize="13")
g.edge("open", "human")
g.edge("welcome", "human")

# ---------------- Per-turn intent flow ----------------
with g.subgraph(name="cluster_say") as c:
    c.attr(label="KgIntentChatSession.say(): per-turn intent flow", style="dashed,filled",
           fillcolor="#f7faf7", color="#b9d2bb", fontname="Helvetica-Oblique", fontsize="11",
           fontcolor="#3d6b40", labeljust="l")
    decision(c, "d_conf", "Pending yes/no\nconfirmation\n(who-gap asked\nlast turn)?")
    decision(c, "d_int", "Pending intent\nanswer (what /\nhow much / where)?")
    decision(c, "d_q", "Question about\nown record / life?\n(\"what did I drink?\")")
    c.node("srl", "SRL extraction -> push new activities to KG\non_new_subject -> tracker.reported[type] += 1",
           fillcolor=C_ACT, group="spine")
    decision(c, "d_gap", "New activity whose\nintent still has\nunmet requirements?")

    c.node("a_conf", tagged("Confirm / deny / correct\n-> push to KG, acknowledge", "gap"), fillcolor=C_ACT)
    c.node("a_int", tagged("Classify & push the answer,\nacknowledge", "gap"), fillcolor=C_ACT)
    decision(c, "d_more", "That activity's\nintent still has\nunmet requirements?")
    c.node("a_more", tagged("Ask its next requirement", "gap"), fillcolor=C_ACT)
    c.node("a_sum", "Intent done: ONE sentence\nsumming up that activity,\nthen agent_fn (next topic)",
           fillcolor=C_ACT)
    c.node("a_q", tagged("Answer from the KG (nothing\nextracted or pushed); a long one\nshortened + \"Want the details?\"\n"
                         "\"summarise my life\": short summary\nof the whole timeline",
                         "kg_answer"), fillcolor=C_ACT)
    c.node("a_gap", tagged("Ask follow-up gap question\nabout THAT activity", "gap"), fillcolor=C_ACT)

# ---------------- Catch-up / saturation layer ----------------
with g.subgraph(name="cluster_catch") as c:
    c.attr(label="wrap_agent_fn_with_saturation_loop(): catch-up / saturation layer", style="dashed,filled",
           fillcolor="#fdf6f0", color="#e2bc9c", fontname="Helvetica-Oblique", fontsize="11",
           fontcolor="#9a5a2a", labeljust="l")
    decision(c, "d_recap", "Human's own last\nmessage asks for\na recap? (\"what have\nwe discussed?\")")
    decision(c, "d_sat", "Tracker saturated?\ntargets met / asked 3x\nAND every topic\nhas come up")
    decision(c, "d_wrap", "Wrap-up\nalready sent?")
    c.node("a_recap", "Recap of what was covered\n(no tracker change)", fillcolor=C_CATCH)
    c.node("a_next", tagged("next_question(): catch-up target\nwith the biggest shortfall; then\n"
                            "EXPLORE each topic not yet\ndiscussed: most recent for this\nperson first, else random",
                            "default"), fillcolor=C_CATCH)
    c.node("a_wrap", tagged("Wrap-up message (once):\nLLM continues briefly\nor says goodbye", "default"),
           fillcolor=C_CATCH)
    c.node("a_llm", tagged("Plain LLM reply\n(ordinary chat)", "default"), fillcolor=C_ACT, group="spine")

g.node("reply", "Agent reply -> annotate + push agent turn to KG\n(skipped for kg_answer)",
       fillcolor=C_IO, fontname="Helvetica-Bold", group="spine")

# Decision and its "yes" action on the same row.
for d, a in (("d_conf", "a_conf"), ("d_int", "a_int"), ("d_q", "a_q"), ("d_gap", "a_gap"),
             ("d_recap", "a_recap"), ("d_sat", "a_next"), ("d_wrap", "a_wrap")):
    with g.subgraph() as r:
        r.attr(rank="same")
        r.node(d)
        r.node(a)

# Main spine.
g.edge("human", "d_conf", weight="20")
g.edge("d_conf", "d_int", label=" no", weight="20")
g.edge("d_int", "d_q", label=" no / unrelated", weight="20")
g.edge("d_q", "srl", label=" no", weight="20")
g.edge("srl", "d_gap", weight="20")
g.edge("d_gap", "d_recap", label=" no -> agent_fn", weight="20")
g.edge("d_recap", "d_sat", label=" no", weight="20")
g.edge("d_sat", "d_wrap", label=" yes", weight="20")
g.edge("d_wrap", "a_llm", label=" yes", weight="20")

# Branches.
g.edge("d_conf", "a_conf", label="yes")
g.edge("d_int", "a_int", label="yes")
g.edge("a_int", "d_more")
g.edge("a_conf", "d_more", style="dashed")
g.edge("d_more", "a_more", label="yes")
g.edge("d_more", "a_sum", label="no")
g.edge("a_sum", "d_recap", label=" agent_fn", style="dashed", constraint="false", fontname="Helvetica",
       fontsize="9", fontcolor="#666666")
g.edge("d_q", "a_q", label="yes")
g.edge("d_gap", "a_gap", label="yes")
g.edge("d_recap", "a_recap", label="yes")
g.edge("d_sat", "a_next", label="no")
g.edge("d_wrap", "a_wrap", label="no")
g.edge("a_next", "a_llm", label=" no topic left (None)", style="dashed", constraint="false", fontname="Helvetica", fontsize="9",
       fontcolor="#666666")

# Every reply -> agent reply -> next turn.
for a in ("a_more", "a_q", "a_gap", "a_recap", "a_next", "a_wrap", "a_llm"):
    g.edge(a, "reply")
# "Next turn" loop, routed up the left via two invisible waypoints on the reply's and human's ranks.
LOOP = dict(color="#7a6aa8", penwidth="1.6")
g.node("lp_bottom", "", shape="point", width="0", height="0", style="invis")
g.node("lp_top", "", shape="point", width="0", height="0", style="invis")
with g.subgraph() as r:
    r.attr(rank="same")
    r.node("lp_bottom")
    r.node("reply")
with g.subgraph() as r:
    r.attr(rank="same")
    r.node("lp_top")
    r.node("human")
g.edge("lp_bottom", "reply", dir="none", headport="w", minlen="3", **LOOP)
g.edge("lp_top", "lp_bottom", dir="none", weight="0", xlabel="next turn ", fontcolor="#7a6aa8", **LOOP)
g.edge("lp_top", "human", headport="w", minlen="3", **LOOP)

# ---------------- Legend ----------------
with g.subgraph(name="cluster_legend") as c:
    c.attr(label="", style="invis")
    c.attr("node", fontsize="10", margin="0.1,0.04", height="0.3")
    items = [("l1", "name / memory query", C_SETUP, "box"), ("l2", "decision", C_DEC, "diamond"),
             ("l3", "base intent-chat step", C_ACT, "box"), ("l4", "catch-up / saturation step", C_CATCH, "box"),
             ("l5", "turn", C_IO, "box"),
             ("l6", "[tag] = kg_session.reply_sources", "white", "plaintext")]
    with c.subgraph() as r:
        r.attr(rank="same")
        for n, t, col, shape in items:
            r.node(n, t, fillcolor=col, shape=shape,
                   style="filled" if shape != "box" else "filled,rounded")
    for (a, *_), (b, *_) in zip(items, items[1:]):
        c.edge(a, b, style="invis")
g.edge("reply", "l3", style="invis")

g.render(OUT, format="svg", cleanup=True)
g.attr(dpi="150")
g.render(OUT, format="png", cleanup=True)
g.save(str(OUT) + ".gv")
print(f"{OUT}.png\n{OUT}.svg\n{OUT}.gv")
