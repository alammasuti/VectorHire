import sys
from pathlib import Path

from fpdf import FPDF
from openai import OpenAI
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import OPENAI_API_KEY, LLM_MODEL
from app.database import get_engine

OUTPUT_DIR = Path(__file__).parent.parent / "resumes"

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

RESUME_PROMPT = """Write a realistic resume for this job candidate, using ONLY the facts given below.
Invent plausible company names and a job history that fits the given years of experience, but never
contradict the facts provided (role, skills, experience, location).

Candidate facts:
- Name: {name}
- Position applied for: {position}
- Skills: {skills}
- Total experience: {total_exp} years
- Relevant experience: {relevant_exp} years
- Location: {location}

Write in plain text (no markdown headers), with these sections:
SUMMARY: 2-3 sentences.
EXPERIENCE: 2-3 bullet points per job, across enough jobs to add up to the total experience.
SKILLS: a short restated list.
EDUCATION: one plausible degree relevant to the role.

Keep it concise, under 300 words."""


def generate_resume_text(row) -> str:
    prompt = RESUME_PROMPT.format(
        name=row.Name,
        position=row.PositionAppliedFor,
        skills=row.Skills,
        total_exp=row.TotalExperience,
        relevant_exp=row.RelevantExperience,
        location=row.LocationCityState,
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def build_pdf_bytes(name: str, resume_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", size=16)
    pdf.multi_cell(0, 10, name)
    pdf.set_font("Helvetica", size=11)
    pdf.ln(2)
    cleaned = resume_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, cleaned)
    return bytes(pdf.output())


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT Id, Name, PositionAppliedFor, Skills, TotalExperience, "
                "RelevantExperience, LocationCityState, ResumeFileName FROM JobApplication "
                "WHERE ResumeFileData IS NULL"
            )
        ).fetchall()

        if not rows:
            print("All candidates already have a resume. Nothing to do.")
            return

        print(f"Generating resume PDFs for {len(rows)} candidates...\n")
        for row in rows:
            resume_text = generate_resume_text(row)
            pdf_bytes = build_pdf_bytes(row.Name, resume_text)

            file_name = row.ResumeFileName or f"candidate_{row.Id}_resume.pdf"
            file_path = OUTPUT_DIR / file_name
            file_path.write_bytes(pdf_bytes)

            conn.execute(
                text("UPDATE JobApplication SET ResumeFileData = :data WHERE Id = :id"),
                {"data": pdf_bytes, "id": row.Id},
            )
            conn.commit()
            print(f"[{row.Id}] {row.Name} -> {file_path} ({len(pdf_bytes)} bytes)")

        print("\nDone.")


if __name__ == "__main__":
    main()
