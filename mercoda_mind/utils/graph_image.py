from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def save_graph(app, name: str) -> Path:
    """Save a compiled LangGraph as docs/<name>.png (needs internet).
    If that fails, save it as Mermaid text (docs/<name>.mmd) instead."""
    DOCS_DIR.mkdir(exist_ok=True)
    graph = app.get_graph()
    try:
        path = DOCS_DIR / f"{name}.png"
        path.write_bytes(graph.draw_mermaid_png())
        print(f"Saved graph image: {path}")
    except Exception as e:
        path = DOCS_DIR / f"{name}.mmd"
        path.write_text(graph.draw_mermaid(), encoding="utf-8")
        print(f"Could not draw the PNG ({type(e).__name__}), saved Mermaid text instead: {path}")
    return path