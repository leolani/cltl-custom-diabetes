"""Links into the GraphDB Workbench for a knowledge graph subject, dependency-free (re/urllib only).

Used by notebooks/kg_chat_gui.py's "Open in GraphDB" button and by cltl.diabetes, which returns a
link to the activities extracted from each utterance alongside the agent's reply.
"""
import re
import urllib.parse

# Matches a GraphDB repository's SPARQL endpoint, e.g.
# "http://localhost:7200/repositories/event_sandbox" -> ("http://localhost:7200", "event_sandbox").
GRAPHDB_REPOSITORY_URL_PATTERN = re.compile(r"^(https?://[^/]+)/repositories/([^/?#]+)/?$")


def graphdb_base_and_repository(kg_address):
    """Split a repository SPARQL endpoint URL into (workbench_base_url, repository_id), e.g.
    "http://localhost:7200/repositories/event_sandbox" -> ("http://localhost:7200",
    "event_sandbox") -- the two pieces graphdb_visualization_url() needs to build a GraphDB
    Workbench link. Returns (None, None) if `kg_address` is falsy or doesn't look like a GraphDB
    repository endpoint (e.g. a different SPARQL 1.1 server); callers use that to skip building
    the graph panel/links instead of producing a broken URL."""
    if not kg_address:
        return None, None
    match = GRAPHDB_REPOSITORY_URL_PATTERN.match(kg_address.strip())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def graphdb_visualization_url(kg_address, subject_uri, role="context", embedded=False):
    """The GraphDB Workbench "Visual graph" URL centered on `subject_uri` -- the same shape
    GraphDB's own UI links to when you open a resource's graph view, e.g.
    "http://localhost:7200/graphs-visualizations?uri=<subject_uri>&role=context&repositoryId=<repo>".
    Returns None if `kg_address` doesn't look like a GraphDB repository endpoint (see
    graphdb_base_and_repository()).

    `embedded` appends GraphDB's bare "&embedded" flag, which shows the graph without the
    Workbench's menus and header, so the page can be embedded, e.g. in the chat UI."""
    base, repository_id = graphdb_base_and_repository(kg_address)
    if not base or not subject_uri:
        return None
    query = urllib.parse.urlencode({"uri": subject_uri, "role": role, "repositoryId": repository_id})
    return f"{base}/graphs-visualizations?{query}" + ("&embedded" if embedded else "")
