"""Draw the dialogue decision process of notebooks/kg_catchup_intent_chat.ipynb as a Graphviz flowchart.

Follows KgIntentChatSession.say() (src/cltl/kg_chat/chat_sessions.py) and
wrap_agent_fn_with_saturation_loop() / SaturationTracker (src/cltl/kg_chat/catch_up_from_kg.py).

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
    c.attr(label="Before the chat (queries on the KG)", style="dashed,filled", fillcolor="#f3f6fa",
           color="#b8c4d6", fontname="Helvetica-Oblique", fontsize="11", labeljust="l")
    c.node("last", "Find last conversation date\nwith HUMAN\n(fallback: 7 days ago)", fillcolor=C_SETUP)
    c.node("topics", "Find catch-up topics:\nweekly rate per activity type\n-> target for the period\n(period capped at 14 days)",
           fillcolor=C_SETUP)
    c.node("open", "Opening turn:\ntime since last chat, summary of\npast topics, question on the\nmost under-covered topic",
           fillcolor=C_CATCH)
    with c.subgraph() as r:
        r.attr(rank="same")
        for n in ("last", "topics", "open"):
            r.node(n)
    c.edge("last", "topics")
    c.edge("topics", "open")

g.node("human", "Human utterance", fillcolor=C_IO, group="spine", fontname="Helvetica-Bold", fontsize="13")
g.edge("open", "human")

# ---------------- Per-turn intent flow ----------------
with g.subgraph(name="cluster_say") as c:
    c.attr(label="KgIntentChatSession.say(): per-turn intent flow", style="dashed,filled",
           fillcolor="#f7faf7", color="#b9d2bb", fontname="Helvetica-Oblique", fontsize="11",
           fontcolor="#3d6b40", labeljust="l")
    decision(c, "d_conf", "Pending yes/no\nconfirmation\n(who-gap asked\nlast turn)?")
    decision(c, "d_int", "Pending intent\nanswer (what /\nhow much / where)?")
    decision(c, "d_q", "Question about\nown record?\n(\"what did I drink?\")")
    c.node("srl", "SRL extraction -> push new activities to KG\non_new_subject -> tracker.reported[type] += 1",
           fillcolor=C_ACT, group="spine")
    decision(c, "d_gap", "New activity whose\nintent still has\nunmet requirements?")

    c.node("a_conf", tagged("Confirm / deny / correct\n-> push to KG, acknowledge", "gap"), fillcolor=C_ACT)
    c.node("a_int", tagged("Classify & push the answer,\nacknowledge / next requirement", "gap"), fillcolor=C_ACT)
    c.node("a_q", tagged("Answer from the KG\n(nothing extracted or pushed)", "kg_answer"), fillcolor=C_ACT)
    c.node("a_gap", tagged("Ask follow-up gap question\nabout THAT activity", "gap"), fillcolor=C_ACT)

# ---------------- Catch-up / saturation layer ----------------
with g.subgraph(name="cluster_catch") as c:
    c.attr(label="wrap_agent_fn_with_saturation_loop(): catch-up / saturation layer", style="dashed,filled",
           fillcolor="#fdf6f0", color="#e2bc9c", fontname="Helvetica-Oblique", fontsize="11",
           fontcolor="#9a5a2a", labeljust="l")
    decision(c, "d_recap", "Human asks\nfor a recap?\n(\"what have we\ndiscussed?\")")
    decision(c, "d_sat", "Tracker saturated?\nevery topic:\nreported >= target\nor asked 3x")
    decision(c, "d_wrap", "Wrap-up\nalready sent?")
    c.node("a_recap", "Recap of what was covered\n(no tracker change)", fillcolor=C_CATCH)
    c.node("a_next", tagged("next_question(): topic with the\nbiggest shortfall (follow-up\nphrasing if asked before)",
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
g.edge("d_q", "a_q", label="yes")
g.edge("d_gap", "a_gap", label="yes")
g.edge("d_recap", "a_recap", label="yes")
g.edge("d_sat", "a_next", label="no")
g.edge("d_wrap", "a_wrap", label="no")
g.edge("a_next", "a_llm", label=" no topic left (None)", style="dashed", constraint="false", fontname="Helvetica", fontsize="9",
       fontcolor="#666666")

# Every reply -> agent reply -> next turn.
for a in ("a_conf", "a_int", "a_q", "a_gap", "a_recap", "a_next", "a_wrap", "a_llm"):
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
    items = [("l1", "setup query", C_SETUP, "box"), ("l2", "decision", C_DEC, "diamond"),
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
