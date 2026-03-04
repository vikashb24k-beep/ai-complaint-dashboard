"""Shared orchestration pipeline used by CLI, API, and dashboard."""

from __future__ import annotations

from typing import Dict

from backend.complaint_ingestion import load_sample_complaints, normalize_input_record
from backend.duplicate_detection import find_similar_complaints
from backend.nlp_analysis import analyze_complaint
from backend.response_generator import generate_response_draft
from backend.root_cause_analysis import infer_root_cause
from backend.sla_tracking import assign_sla_hours
from database.complaint_db import ComplaintDB

DEFAULT_DB_PATH = "database/complaints.db"
DEFAULT_SAMPLE_PATH = "data/sample_complaints.csv"


def process_single_complaint(db: ComplaintDB, record: Dict) -> int:
    normalized = normalize_input_record(record)
    existing = db.fetch_complaints()
    existing_for_similarity = [
        {"id": row["id"], "complaint_text": row["complaint_text"]} for row in existing
    ]

    complaint_id = db.insert_complaint(normalized)
    db.add_timeline_event(complaint_id, "INGESTED", "Complaint ingested from channel")

    analysis = analyze_complaint(normalized["complaint_text"])
    root_cause = infer_root_cause(normalized["complaint_text"])
    sla_hours = assign_sla_hours(analysis["severity"], analysis["category"])
    db.update_analysis(
        complaint_id=complaint_id,
        category=analysis["category"],
        product=analysis["product"],
        sentiment=analysis["sentiment"],
        severity=analysis["severity"],
        key_issues=analysis["key_issues"],
        summary=analysis["summary"],
        root_cause=root_cause,
        sla_hours=sla_hours,
        escalated=int(analysis["severity"] in {"high", "critical"}),
    )
    db.add_timeline_event(complaint_id, "ANALYZED", "NLP and severity analysis completed")

    duplicates = find_similar_complaints(
        normalized["complaint_text"],
        existing_for_similarity,
        threshold=0.72,
        top_k=3,
    )
    if duplicates:
        best = duplicates[0]
        db.update_duplicate_link(complaint_id, best["id"], best["similarity"])
        db.add_timeline_event(
            complaint_id,
            "DUPLICATE_DETECTED",
            f"Potential duplicate of complaint #{best['id']} ({best['similarity']:.2f})",
        )

    response = generate_response_draft(
        complaint_text=normalized["complaint_text"],
        category=analysis["category"],
        product=analysis["product"],
        sentiment=analysis["sentiment"],
        severity=analysis["severity"],
        key_issues=analysis["key_issues"],
    )
    db.update_response(complaint_id, response["draft"], response["mode"])
    db.add_timeline_event(complaint_id, "RESPONSE_DRAFTED", "GenAI response draft generated")

    if analysis["severity"] in {"high", "critical"}:
        db.escalate_complaint(complaint_id)
        db.add_timeline_event(complaint_id, "ESCALATED", "Auto-escalated due to severity")

    return complaint_id


def bootstrap_from_sample(
    db_path: str = DEFAULT_DB_PATH,
    sample_csv: str = DEFAULT_SAMPLE_PATH,
    reset: bool = False,
) -> ComplaintDB:
    db = ComplaintDB(db_path)
    db.initialize()
    if reset:
        db.reset_data()
    if db.count_complaints() == 0:
        sample_df = load_sample_complaints(sample_csv)
        for row in sample_df.to_dict(orient="records"):
            process_single_complaint(db, row)
    return db
