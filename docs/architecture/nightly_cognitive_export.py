# nightly_cognitive_export.py (ecosystem pattern)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nightly_cognitive_export")

class DataLakeError(Exception):
    pass

def _load_parquet_robust(path, required=False):
    """Ecosystem standard for Parquet handling."""
    if not path.exists():
        if required:
            raise DataLakeError(f"Required Parquet missing: {path}")
        logger.warning(f"Optional Parquet missing: {path} — using fallback")
        return None  # or empty structure
    try:
        # actual pq.read_table...
        return "loaded_data"  # placeholder
    except Exception as e:
        logger.error(f"Corrupt Parquet {path}: {e}")
        if required:
            raise DataLakeError from e
        return None

def generate_obsidian_note(date_str, training_signal, obsidian_vault, sources, recent_activity, missing_sources=None):
    if missing_sources is None:
        missing_sources = []
    
    note_dir = obsidian_vault / "Daily" / "Cognitive"
    note_dir.mkdir(parents=True, exist_ok=True)
    note_path = note_dir / f"{date_str}.md"

    frontmatter = f"""---
 date: {date_str}
 tags: [ecosystem/cognitive, plasticity]
 modulation: {training_signal.get('modulation', 0)}
 ---
"""

    lines = [f"# Cognitive Intelligence — {date_str}\n\n"]
    
    if missing_sources:
        lines.append("## ⚠️ Data Sources Status\n")
        for src in missing_sources:
            lines.append(f"- {src}: **MISSING** — fallback data used. Run consolidation job.\n")
        lines.append("\n")
    
    lines.append("## Training Signal\n")
    # ... rest of note content ...
    
    if missing_sources:
        lines.append("\n> Note: Some data sources were unavailable. Full cross-domain knowledge graph links may be limited until next successful run.\n")
    
    note_path.write_text(frontmatter + "".join(lines), encoding="utf-8")
    return note_path

# Usage in run_job: 
# missing = []
# for src in ["juniorclimbs", "juniorstock"]:
#     data = _load_parquet_robust(...)
#     if data is None: missing.append(src)
# generate_obsidian_note(..., missing_sources=missing)