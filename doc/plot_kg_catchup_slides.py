"""Slide-sized flowcharts of the kg_catchup_intent_chat.ipynb dialogue decision process.

The full process is in doc/kg_catchup_decision_flow.png (plot_kg_catchup_decision_flow.py); that is too
big for one slide, so this draws it as an overview plus one chart per main step, left to right for 16:9:

    0_overview      the main steps and how a turn moves through them
    1_setup         before the catch-up: ask the name, memory from the KG, opening or welcome
    2_intent_turn   KgIntentChatSession.say(): the per-turn intent flow
    3_catch_up      wrap_agent_fn_with_saturation_loop(): the catch-up / saturation layer

Needs the Graphviz binaries (`brew install graphviz`) and the `graphviz` Python package.

    .venv/bin/python doc/plot_kg_catchup_slides.py   ->  doc/kg_catchup_slides/*.png and *.svg
"""
from pathlib import Path

import graphviz

OUT_DIR = Path(__file__).with_name("kg_catchup_slides")

C_SETUP = "#dbe4f0"
C_DEC = "#fff2c2"
C_ACT = "#d8efd9"
C_CATCH = "#fbdcc4"
C_IO = "#e6e0f3"
C_NOTE = "#ffffff"
C_LOOP = "#7a6aa8"

def html(text, tag=None, title=None):
    """HTML-like label: optional bold title line, body lines, optional small reply_sources tag."""
    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = []
    if title:
        parts.append(f"<B>{esc(title)}</B>")
    if text:
        parts.extend(esc(line) for line in text.split("\n"))
    body = "<BR/>".join(parts)
    if tag:
        body += f'<BR/><FONT FACE="Courier" POINT-SIZE="11" COLOR="#666666">[{tag}]</FONT>'
    return f"<{body}>"


def new_graph(name, title):
    g = graphviz.Digraph(name)
    g.attr(rankdir="LR", newrank="true", nodesep="0.35", ranksep="0.55", fontname="Helvetica",
           label=f"{title}\n ", labelloc="t", fontsize="24", bgcolor="white", pad="0.3")
    g.attr("node", fontname="Helvetica", fontsize="14", style="filled,rounded", shape="box",
           color="#444444", penwidth="1.3", margin="0.18,0.1")
    g.attr("edge", fontname="Helvetica-Bold", fontsize="13", color="#444444", arrowsize="0.9", penwidth="1.3")
    return g


def decision(sg, name, text):
    sg.node(name, text, shape="diamond", style="filled", fillcolor=C_DEC, margin="0.04", group="spine")


def same_rank(g, *nodes):
    with g.subgraph() as r:
        r.attr(rank="same")
        for n in nodes:
            r.node(n)


def render(g, name):
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / name
    g.render(out, format="svg", cleanup=True)
    g.attr(dpi="200")
    g.render(out, format="png", cleanup=True)
    print(f"{out}.png")


# --------------------------------------------------------------------------- #
# 0. Overview
# --------------------------------------------------------------------------- #
def overview():
    g = new_graph("0_overview", "KG catch-up chat: main steps")
    g.attr(ranksep="0.7", nodesep="0.6")

    g.node("setup", html("ask the name (known people from the KG)\nmemory: last conversation,\n"
                         "catch-up topics + targets, last topics\nopening turn, or a welcome",
                         title="1  Who is it?"), fillcolor=C_SETUP)
    g.node("human", "Human\nutterance", fillcolor=C_IO, fontname="Helvetica-Bold", fontsize="16")
    g.node("intent", html("answer a pending question\nanswer a question from the KG\n"
                          "extract + push activities to KG,\nask intent follow-ups;\n"
                          "intent done: one-sentence summary",
                          title="2  Per-turn intent flow"), fillcolor=C_ACT)
    g.node("catch", html("recap on request\nask the next under-covered topic,\n"
                         "then every topic not yet discussed\nwrap up once saturated\nthen: ordinary chat",
                         title="3  Catch-up / saturation layer"), fillcolor=C_CATCH)
    g.node("reply", "Agent reply\n(pushed to KG)", fillcolor=C_IO, fontname="Helvetica-Bold", fontsize="16")

    g.edge("setup", "human", label=" opening turn")
    g.edge("human", "intent")
    g.edge("intent", "catch", label=" no follow-up\n left")
    g.node("llm", html("OpenAI, or a local llama-server\n(llm_backend; LLM_BACKEND\nin the notebook)",
                       title="LLM"), shape="note", style="filled", fillcolor=C_NOTE, fontsize="12",
           color="#999999")
    g.edge("reply", "llm", style="invis")
    g.edge("catch", "reply")
    g.edge("intent", "reply", label=" reply found", constraint="false")
    g.edge("reply", "human", label="next turn", color=C_LOOP, fontcolor=C_LOOP, penwidth="1.8",
           constraint="false", tailport="s", headport="s")

    render(g, "0_overview")


# --------------------------------------------------------------------------- #
# 1. Before the chat
# --------------------------------------------------------------------------- #
def setup():
    g = new_graph("1_setup", "Step 1: who is it, and what does the KG remember?")
    g.attr(ranksep="0.45")

    g.node("known", html("everyone the agent has\ntalked with (KG) +\n\"What's your name?\"",
                         title="Opening"), fillcolor=C_CATCH)
    g.node("name", html("a known name (any case),\nelse the LLM picks it out;\nno name: ask again",
                        title="Name"), fillcolor=C_SETUP)
    decision(g, "d_known", "Known\nperson?")
    g.node("memory", html("last conversation\ntopics of the last chat\n(their own words)\n"
                          "when each topic was\nlast discussed\n-> timeline in the GUI",
                          title="Memory"), fillcolor=C_SETUP)
    g.node("target", html("period = min(gap, 14 days)\nweekly rate per type ->\n"
                          "target = max(1, round(\n  rate x period days / 7))",
                          title="Catch-up targets"), fillcolor=C_SETUP)
    g.node("open", html("1. time since last chat\n2. topics of the LAST chat\n"
                        "3. question on the topic with\n    the biggest shortfall",
                        title="Opening turn"), fillcolor=C_CATCH)
    g.node("welcome", html("no catch-up topics,\nno gap; ask how\nthey have been",
                           title="Welcome (first meeting)"), fillcolor=C_CATCH)
    g.node("human", "Human\nutterance", fillcolor=C_IO, fontname="Helvetica-Bold", fontsize="16")

    # Snake: name and memory left-to-right on top, then right-to-left underneath.
    g.attr(rankdir="TB")
    same_rank(g, "known", "name", "d_known", "memory")
    same_rank(g, "human", "welcome", "open", "target")
    for a, b in (("known", "name"), ("name", "d_known")):
        g.edge(a, b)
    g.edge("d_known", "memory", label=" yes")
    g.edge("memory", "target")
    # Flat edges place their tail on the left, so draw these reversed with dir="back".
    g.edge("open", "target", dir="back")
    g.edge("human", "open", dir="back")
    g.edge("d_known", "welcome", label=" no")
    g.edge("welcome", "human", style="invis")
    g.edge("human", "welcome", dir="back")
    render(g, "1_setup")


# --------------------------------------------------------------------------- #
# 2. Per-turn intent flow
# --------------------------------------------------------------------------- #
def rows(g, spine, branches, reply="reply", extra_bottom=()):
    """Three rows: the decision spine left to right, each decision's action under it, the reply below."""
    g.attr(rankdir="TB", nodesep="0.3", ranksep="0.6")
    same_rank(g, *[n for n, _, _ in spine], spine[-1][1])
    for a, b, lab in spine:
        g.edge(a, b, label=f" {lab} " if lab else "")
    same_rank(g, *[a for _, a, _ in branches])
    for d, a, lab in branches:
        g.edge(d, a, label=f" {lab}", weight="5")
        g.edge(a, reply)
    same_rank(g, reply, *extra_bottom)


def intent_turn():
    g = new_graph("2_intent_turn", "Step 2: per-turn intent flow (KgIntentChatSession.say())")

    g.node("human", "Human\nutterance", fillcolor=C_IO, fontname="Helvetica-Bold", fontsize="16")
    decision(g, "d_conf", "Pending\nyes/no\nconfirmation?")
    decision(g, "d_int", "Pending\nintent answer?\n(what / how\nmuch / where)")
    decision(g, "d_q", "Question\nabout own\nrecord?")
    g.node("srl", html("SRL extraction\npush new activities\ntracker.reported += 1",
                       title="Extract"), fillcolor=C_ACT)
    decision(g, "d_gap", "New activity\nwith unmet\nintent\nrequirements?")
    g.node("catch", html("no follow-up left:\ncall agent_fn", title="-> Step 3"), fillcolor=C_CATCH)

    g.node("a_conf", html("confirm / deny /\ncorrect -> push\n+ acknowledge", tag="gap"), fillcolor=C_ACT)
    g.node("a_int", html("classify answer\n-> push; next\nrequirement, or\nintent done:\n"
                         "1-sentence summary\n-> step 3", tag="gap"), fillcolor=C_ACT)
    g.node("a_q", html("answer from\nthe KG (no push);\nlong: gist +\n\"Want the\ndetails?\"\n"
                       "life: short summary\nof the timeline",
                       tag="kg_answer"), fillcolor=C_ACT)
    g.node("a_gap", html("follow-up\nquestion about\nTHAT activity", tag="gap"), fillcolor=C_ACT)
    g.node("reply", "Agent reply", fillcolor=C_IO, fontname="Helvetica-Bold", fontsize="16")
    g.node("note", html("A who-gap (agent role) is asked as a yes/no confirmation;\n"
                        "a what / how much / where gap arms a pending intent answer.\n"
                        "Each intent requirement is asked a limited number of times.\n"
                        "When an activity's intent has nothing left to ask, its acknowledgement\n"
                        "becomes one sentence about that activity only, and step 3 asks the next topic.",
                        title="Follow-up questions"),
           shape="note", style="filled", fillcolor=C_NOTE, fontsize="12", color="#999999")

    rows(g,
         spine=[("human", "d_conf", ""), ("d_conf", "d_int", "no"), ("d_int", "d_q", "no / unrelated"),
                ("d_q", "srl", "no"), ("srl", "d_gap", ""), ("d_gap", "catch", "no")],
         branches=[("d_conf", "a_conf", "yes"), ("d_int", "a_int", "yes"), ("d_q", "a_q", "yes"),
                   ("d_gap", "a_gap", "yes")],
         extra_bottom=("note",))
    g.edge("reply", "note", style="invis", minlen="2")
    render(g, "2_intent_turn")


# --------------------------------------------------------------------------- #
# 3. Catch-up / saturation layer
# --------------------------------------------------------------------------- #
def catch_up():
    g = new_graph("3_catch_up", "Step 3: catch-up / saturation layer (wrap_agent_fn_with_saturation_loop())")

    g.node("from", html("no intent follow-up\nleft this turn", title="From step 2"), fillcolor=C_ACT)
    decision(g, "d_recap", "Human's\nmessage asks\nfor a recap?")
    decision(g, "d_sat", "Tracker\nsaturated?")
    decision(g, "d_wrap", "Wrap-up\nalready\nsent?")
    g.node("a_llm", html("plain LLM reply\n(ordinary chat)", tag="default"), fillcolor=C_ACT)

    g.node("a_recap", html("recap of what\nwas covered\n(no tracker change)"), fillcolor=C_CATCH)
    g.node("a_next", html("next_question():\n1. catch-up target with\nthe biggest shortfall\n"
                          "2. explore a topic not yet\ndiscussed: most recent\nfor them first, else random",
                          tag="default"), fillcolor=C_CATCH)
    g.node("a_wrap", html("wrap-up message (once):\nLLM continues briefly\nor says goodbye", tag="default"),
           fillcolor=C_CATCH)
    g.node("reply", "Agent reply", fillcolor=C_IO, fontname="Helvetica-Bold", fontsize="16")
    g.node("note", html("A catch-up target is done when reported >= target or it was asked 3 times.\n"
                        "The tracker is saturated when every target is done AND every topic of the\n"
                        "overall list has come up (reported, asked, or explored once).\n"
                        "reported is updated live by on_new_subject (step 2, Extract).",
                        title="Saturation"),
           shape="note", style="filled", fillcolor=C_NOTE, fontsize="12", color="#999999")

    rows(g,
         spine=[("from", "d_recap", ""), ("d_recap", "d_sat", "no"), ("d_sat", "d_wrap", "yes"),
                ("d_wrap", "a_llm", "yes")],
         branches=[("d_recap", "a_recap", "yes"), ("d_sat", "a_next", "no"), ("d_wrap", "a_wrap", "no")],
         extra_bottom=("note",))
    g.edge("a_llm", "reply")
    g.edge("a_next", "a_llm", label=" no topic left ", style="dashed", constraint="false",
           fontname="Helvetica", fontsize="11", fontcolor="#666666", color="#888888")
    g.edge("reply", "note", style="invis", minlen="2")
    render(g, "3_catch_up")


if __name__ == "__main__":
    overview()
    setup()
    intent_turn()
    catch_up()
