#!/usr/bin/env python3
"""
Redrob Hackathon – Final Ranking Engine (v9.0)
Team: The defenders | Solo: Tanish M

Pipeline:
  Phase 1   – Hard drops (country, services-only, product-months, non-coding, YOE)
  Phase 1B  – Honeypot detection (score * 0.01 multiplier)
  Phase 2A  – Multiplicative penalties (incl. availability/activity down-weights)
  Phase 2B  – Additive boosts (incl. TF-IDF semantic similarity to the JD)
  Phase 3   – Dynamic reasoning for top-100

Usage:
  python rank.py --candidates candidates.jsonl --out team_the_defenders.csv
"""

import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# JOB DESCRIPTION TEXT (for semantic-similarity scoring, Phase 2B)
# Condensed from job_description.docx — core requirements only, hackathon
# meta-commentary excluded so the vector represents the actual role.
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer, Founding Team, Redrob AI. Own the intelligence layer:
ranking, retrieval, and matching systems. Deep technical depth in modern ML
systems: embeddings, retrieval, ranking, LLMs, fine-tuning. Scrappy
product-engineering attitude, ships working systems fast.
Production experience with embeddings-based retrieval systems: sentence
transformers, OpenAI embeddings, BGE, E5. Handled embedding drift, index
refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search: Pinecone,
Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python, code quality.
Evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online
correlation, A/B testing.
LLM fine-tuning: LoRA, QLoRA, PEFT. Learning-to-rank models: XGBoost, neural.
HR-tech, recruiting tech, marketplace products. Distributed systems,
large-scale inference optimization. Open-source contributions.
Shipped an end-to-end ranking, search, or recommendation system to real
users at meaningful scale. Hybrid vs dense retrieval, offline vs online
evaluation, when to fine-tune vs prompt.
"""

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SERVICES_INDUSTRIES = {
    "IT Services", "Consulting", "Management Consulting",
    "Outsourcing", "BPO", "IT Consulting"
}

KNOWN_SERVICE_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "l&t infotech",
    "mindtree", "hexaware", "persistent"
}

NON_TECH_TITLES = {
    "marketing", "sales", "hr", "human resources", "recruiter",
    "finance", "accountant", "designer", "graphic designer",
    "operations", "product manager", "business development",
    "business analyst", "content writer", "customer support",
    "civil engineer", "mechanical engineer", "project manager",
    "sales executive"
}

ML_KEYWORDS = {
    "embedding", "embeddings", "retrieval", "ranking", "llm",
    "fine-tuning", "fine tuning", "sentence-transformers",
    "sentence transformers", "pinecone", "milvus", "qdrant",
    "weaviate", "faiss", "opensearch", "elasticsearch",
    "rag", "re-ranking", "reranking", "ndcg", "mrr",
    "mean average precision", "a/b test", "bm25", "hybrid search",
    "dense retrieval", "learning-to-rank", "learning to rank",
    "xgboost", "lightgbm", "recommendation",
    "collaborative filtering", "feature engineering"
}

NLP_IR_SKILLS = {
    "nlp", "information retrieval", "embeddings", "semantic search",
    "ranking", "retrieval", "sentence transformers", "rag",
    "vector search", "bm25", "faiss", "pinecone", "qdrant",
    "milvus", "weaviate", "opensearch", "elasticsearch",
    "hugging face transformers", "fine-tuning llms",
    "recommendation systems", "feature engineering"
}

CV_SKILLS = {
    "computer vision", "opencv", "yolo", "image classification",
    "object detection", "speech recognition", "tts", "asr",
    "robotics", "gans", "cnn"
}

PRODUCTION_TERMS = {
    "shipped", "deployed", "production", "a/b test", "launched",
    "users", "our product", "scale", "serving", "inference",
    "latency", "throughput", "pipeline"
}

CODING_TERMS = {
    "built", "shipped", "deployed", "implemented", "python",
    "model", "architecture", "code", "developed", "engineering",
    "system", "designed", "pipeline"
}

# JD preferred locations: Noida/Pune (ideal) vs other metros (welcome)
IDEAL_LOCATIONS   = {"pune", "noida"}
WELCOME_LOCATIONS = {"hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram"}

NON_CODING_TITLES = {
    "architect", "tech lead", "engineering manager", "head of",
    "director", "vp", "principal"
}

TIER_1_COMPANIES = {
    "google", "meta", "facebook", "amazon", "apple", "netflix", "microsoft",
    "nvidia", "openai", "deepmind"
}

TIER_2_COMPANIES = {
    "uber", "salesforce", "adobe", "linkedin", "twitter", "stripe", "atlassian",
    "airbnb", "dropbox", "spotify", "oracle", "ibm", "sap", "vmware",
    "databricks", "snowflake"
}

TIER_3_COMPANIES = {
    "swiggy", "zomato", "ola", "flipkart", "cred", "razorpay",
    "sarvam ai", "yellow.ai", "haptik", "observe.ai", "paytm", "zoho"
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def days_since(date_str: str, now: datetime) -> int:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (now - d).days

def is_non_tech_title(title: str) -> bool:
    t = title.lower()
    # Allow data/ml/bi analysts through
    if "analyst" in t and any(x in t for x in ["data", "ml", "bi", "business intelligence"]):
        return False
    return any(nt in t for nt in NON_TECH_TITLES)

def is_service_company(company: str) -> bool:
    comp = company.strip().lower()
    if any(firm in comp for firm in KNOWN_SERVICE_FIRMS):
        return True
    if any(kw in comp for kw in ("consulting", "outsourcing", "bpo", "tech services")):
        return True
    return False

def is_tier_company(company: str, tier_set: set) -> bool:
    comp = company.strip().lower()
    return any(tier_firm in comp for tier_firm in tier_set)

# ---------------------------------------------------------------------------
# COUNTING
# ---------------------------------------------------------------------------
def compute_qualifying_months(c: Dict) -> int:
    """Sum all product-company months in technical (non-non-tech) roles."""
    total = 0
    for job in c.get("career_history", []):
        if job.get("industry", "") in SERVICES_INDUSTRIES:
            continue
        if is_service_company(job.get("company", "")):
            continue
        if is_non_tech_title(job.get("title", "")):
            continue
        total += job.get("duration_months", 0)
    return total

def streak_months_to_reward(months: int, tier: int) -> float:
    if months >= 72:
        base = 2.0
    elif months >= 48:
        base = 1.5
    elif months >= 36:
        base = 1.0
    else:
        return 0.0
    if tier == 3:
        return base * 1.5
    elif tier == 2:
        return base * 1.2
    else:
        return base

def tiered_streak_reward(career: List[Dict]) -> float:
    if not career:
        return 0.0
    sorted_jobs = sorted(career, key=lambda j: j.get("start_date", "9999-99-99"))
    best_reward = 0.0
    current_streak_months = 0
    current_streak_tier = 0
    prev_end = None

    for job in sorted_jobs:
        industry = job.get("industry", "")
        company = job.get("company", "").strip().lower()
        is_service = industry in SERVICES_INDUSTRIES or is_service_company(company)
        start_str = job.get("start_date")
        if not start_str:
            continue
        start = datetime.strptime(start_str, "%Y-%m-%d")

        if prev_end is not None:
            gap_days = (start - prev_end).days
            if gap_days > 180:
                if current_streak_months >= 36:
                    best_reward = max(best_reward, streak_months_to_reward(current_streak_months, current_streak_tier))
                current_streak_months = 0
                current_streak_tier = 0

        if is_service:
            if current_streak_months >= 36:
                best_reward = max(best_reward, streak_months_to_reward(current_streak_months, current_streak_tier))
            current_streak_months = 0
            current_streak_tier = 0
        else:
            current_streak_months += job.get("duration_months", 0)
            if is_tier_company(company, TIER_1_COMPANIES):
                current_streak_tier = max(current_streak_tier, 3)
            elif is_tier_company(company, TIER_2_COMPANIES):
                current_streak_tier = max(current_streak_tier, 2)
            elif is_tier_company(company, TIER_3_COMPANIES):
                current_streak_tier = max(current_streak_tier, 1)

        if not job.get("is_current") and job.get("end_date"):
            prev_end = datetime.strptime(job["end_date"], "%Y-%m-%d")
        else:
            prev_end = None

    if current_streak_months >= 36:
        best_reward = max(best_reward, streak_months_to_reward(current_streak_months, current_streak_tier))
    return best_reward

# ---------------------------------------------------------------------------
# PHASE 1: HARD DROPS
# ---------------------------------------------------------------------------
def hard_drop(c: Dict, now: datetime) -> bool:
    profile = c["profile"]
    signals = c["redrob_signals"]
    career = c.get("career_history", [])

    # 1. Not in India
    if profile.get("country") != "India":
        return True

    # availability/activity: soft penalty in Phase 2A, not a hard drop

    # 4. Pure IT services career (zero product company jobs)
    if career:
        product_jobs = sum(
            1 for job in career
            if job.get("industry") not in SERVICES_INDUSTRIES
            and not is_service_company(job.get("company", ""))
        )
        if product_jobs == 0:
            return True

    # 5. Insufficient product-company tech experience (< 48 months)
    if compute_qualifying_months(c) < 48:
        return True

    # 6. Non-coding current role >= 18 months (hard drop)
    for job in career:
        if job.get("is_current"):
            title = job.get("title", "").lower()
            desc = job.get("description", "").lower()
            duration = job.get("duration_months", 0)
            if any(t in title for t in NON_CODING_TITLES) and not any(term in desc for term in CODING_TERMS):
                if duration >= 18:
                    return True
            break

    # 7. Total experience < 4 years
    if profile.get("years_of_experience", 0) < 4:
        return True

    return False

# ---------------------------------------------------------------------------
# PHASE 1B: HONEYPOT DETECTION
# ---------------------------------------------------------------------------
def is_honeypot(c: Dict) -> bool:
    # flag profiles with 2+ impossible skill signals
    profile = c["profile"]
    skills = c.get("skills", [])
    yoe = profile.get("years_of_experience", 0)
    red_flags = 0

    # Expert + 0 months
    if any(s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0 for s in skills):
        red_flags += 1

    # Too many experts for YOE
    high_prof = sum(1 for s in skills if s.get("proficiency") in ("advanced", "expert"))
    if high_prof >= 10 and yoe < 5:
        red_flags += 1

    # Impossible skill duration
    if any(s.get("duration_months", 0) > (yoe * 12) + 24 for s in skills):
        red_flags += 1

    # Un-endorsed experts (zero endorsements on an "expert" claim)
    zero_endorsed_experts = [
        s for s in skills
        if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0
    ]
    if len(zero_endorsed_experts) >= 6:
        red_flags += 1

    return red_flags >= 2

# ---------------------------------------------------------------------------
# PHASE 2A: MULTIPLICATIVE PENALTIES
# ---------------------------------------------------------------------------
def penalty_multiplier(c: Dict, now: datetime) -> float:
    mul = 1.0
    profile = c["profile"]
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    signals = c["redrob_signals"]
    all_descs = " ".join(job.get("description", "").lower() for job in career)

    nlp_ir_months = sum(
        s.get("duration_months", 0) for s in skills if s["name"].lower() in NLP_IR_SKILLS
    )

    # 1. Shallow LLM-only (×0.1)
    has_hype = any(
        s["name"].lower() in ("langchain", "openai", "llamaindex", "prompt engineering")
        and s.get("duration_months", 0) <= 12
        for s in skills
    )
    has_deep = any(
        s["name"].lower() in NLP_IR_SKILLS and s.get("duration_months", 0) > 24
        for s in skills
    )
    has_pre_llm_work = any(
        int(job.get("start_date", "9999")[:4]) < 2023
        and any(kw in job.get("description", "").lower() for kw in ML_KEYWORDS)
        for job in career
    )
    if has_hype and not has_deep and not has_pre_llm_work:
        mul *= 0.1

    # 1b. Framework dominance (×0.3)
    langchain_months = sum(
        s.get("duration_months", 0) for s in skills
        if s["name"].lower() in ("langchain", "llamaindex", "openai")
    )
    if langchain_months > 24 and langchain_months > (nlp_ir_months * 0.5) and not has_deep:
        mul *= 0.3

    # 2. Pure research (×0.1)
    if not any(term in all_descs for term in PRODUCTION_TERMS):
        mul *= 0.1

    # 3. Non-coding current role < 18 months (soft penalty ×0.5)
    for job in career:
        if job.get("is_current"):
            title = job.get("title", "").lower()
            desc = job.get("description", "").lower()
            dur = job.get("duration_months", 0)
            if any(t in title for t in NON_CODING_TITLES) and not any(term in desc for term in CODING_TERMS):
                if dur < 18:
                    mul *= 0.5
            break

    # 4. CV/Speech primary without meaningful NLP/IR depth (×0.3)
    cv_months = sum(
        s.get("duration_months", 0) for s in skills if s["name"].lower() in CV_SKILLS
    )
    if cv_months > 0 and cv_months > nlp_ir_months and nlp_ir_months < 24:
        mul *= 0.3

    # 5. Prompt Engineering dominance (×0.2)
    top_3 = [
        s["name"].lower()
        for s in sorted(skills, key=lambda x: x.get("duration_months", 0), reverse=True)[:3]
    ]
    if "prompt engineering" in top_3 and nlp_ir_months < 48:
        mul *= 0.2

    # 6. Title-chaser / job-hopper (×0.1)
    if len(career) >= 3:
        tenures = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures)
        titles = [j.get("title", "").lower() for j in sorted(career, key=lambda x: x.get("start_date", ""))]
        bumps = 0
        for i in range(1, len(titles)):
            if any(t in titles[i] for t in ("senior", "staff", "principal", "lead")) and \
               not any(t in titles[i - 1] for t in ("senior", "staff", "principal", "lead")):
                bumps += 1
        if avg_tenure < 18 and bumps >= 2:
            mul *= 0.1

    # 7. Junior title (×0.5)
    if "junior" in profile.get("current_title", "").lower():
        mul *= 0.5

    # 8. Notice period pressure
    notice = signals.get("notice_period_days", 60)
    if notice > 120:
        mul *= 0.9
    elif notice > 90:
        mul *= 0.95

    # 9. Service-firm + Junior (double red flag ×0.3)
    curr_comp = profile.get("current_company", "").strip().lower()
    if is_service_company(curr_comp) and "junior" in profile.get("current_title", "").lower():
        mul *= 0.3

    # 10. Zero-retrieval-signal penalty (×0.1)
    retrieval_desc_terms = [
        "embedding", "embeddings", "retrieval", "ranking", "vector search",
        "faiss", "pinecone", "qdrant", "milvus", "weaviate", "elasticsearch",
        "opensearch", "bm25", "learning to rank", "recommendation system"
    ]
    has_retrieval_desc = any(kw in all_descs for kw in retrieval_desc_terms)
    if nlp_ir_months == 0 and not has_retrieval_desc:
        mul *= 0.1

    # 11. Not open to work and no recent applications (×0.2)
    if not signals.get("open_to_work_flag") and signals.get("applications_submitted_30d", 0) == 0:
        mul *= 0.2

    # 12. Inactive > 6 months (×0.15)
    last_active = signals.get("last_active_date")
    if last_active and days_since(last_active, now) > 180:
        mul *= 0.15

    return mul

# ---------------------------------------------------------------------------
# PHASE 2B: ADDITIVE BOOSTS
# ---------------------------------------------------------------------------
def build_candidate_text(c: Dict) -> str:
    """Free-text representation of a candidate for TF-IDF similarity against the JD."""
    profile = c["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    parts.extend(s.get("name", "") for s in c.get("skills", []))
    return " ".join(p for p in parts if p)


def compute_boost_score(c: Dict, qualifying_months: int, jd_similarity: float = 0.0) -> float:
    score = 0.0
    score += jd_similarity * 6.0  # TF-IDF cosine similarity vs JD text
    profile = c["profile"]
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    signals = c["redrob_signals"]
    all_descs = " ".join(job.get("description", "").lower() for job in career)

    current_title = profile.get("current_title", "").lower()
    current_desc = next(
        (j.get("description", "").lower() for j in career if j.get("is_current")), ""
    )

    # Senior title + long avg tenure + hands-on coder (+3.0)
    if any(t in current_title for t in ("senior", "lead", "principal", "staff")):
        if len(career) >= 2:
            avg_tenure = sum(j.get("duration_months", 0) for j in career) / len(career)
            if avg_tenure >= 24 and any(term in current_desc for term in CODING_TERMS):
                score += 3.0

    # Currently hands-on coder (+2.0)
    if any(term in current_desc for term in CODING_TERMS):
        score += 2.0

    # location boost: ideal > welcome > willing to relocate
    location = profile.get("location", "").lower()
    if any(city in location for city in IDEAL_LOCATIONS):
        score += 2.0
    elif any(city in location for city in WELCOME_LOCATIONS):
        score += 1.0
    elif signals.get("willing_to_relocate"):
        score += 0.5

    # Notice period <= 30 days (+1.0)
    if signals.get("notice_period_days", 999) <= 30:
        score += 1.0

    # Recruiter engagement (+1.0)
    if signals.get("recruiter_response_rate", 0) > 0.7 and signals.get("interview_completion_rate", 0) > 0.6:
        score += 1.0

    # Skill assessments (max +2.0)
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        ml_assess = sum(
            1 for k in assessments
            if k.lower() in NLP_IR_SKILLS or k.lower() in ML_KEYWORDS
        )
        score += min(ml_assess * 0.5, 2.0)

    # Verified profile (+0.5)
    if signals.get("verified_email") and signals.get("verified_phone"):
        score += 0.5

    # GitHub activity (max +2.0)
    github = signals.get("github_activity_score", -1)
    if github > 0:
        score += min(github * 0.02, 2.0)

    # ML keywords in career descriptions (max +8.0)
    kw_count = sum(1 for kw in ML_KEYWORDS if kw in all_descs)
    score += min(kw_count, 8.0)

    # Niche valuable skills
    skill_names = {s["name"].lower() for s in skills}
    if any(k in skill_names for k in ("lora", "qlora", "peft")):
        score += 2.0
    if any(k in skill_names for k in ("xgboost", "lightgbm", "learning to rank")):
        score += 1.5

    # Domain experience in HR-Tech / Recruiting (+1.0)
    if any(
        job.get("industry", "").lower() in ("hr-tech", "recruiting", "talent intelligence")
        for job in career
    ):
        score += 1.0

    # Python proficiency
    for s in skills:
        if s["name"].lower() == "python":
            if s.get("proficiency") in ("advanced", "expert"):
                score += 2.0
            elif s.get("proficiency") == "intermediate":
                score += 1.0
            break

    # English proficiency (from root-level languages list)
    for lang in c.get("languages", []):
        if lang.get("language", "").lower() == "english":
            prof = lang.get("proficiency", "").lower()
            if prof == "native":
                score += 1.0
            elif prof == "professional":
                score += 0.5
            break

    # Behavioral signals
    if signals.get("profile_completeness_score", 0) >= 80:
        score += 0.5
    if signals.get("applications_submitted_30d", 0) >= 5:
        score += 0.5
    avg_resp = signals.get("avg_response_time_hours", 999)
    if 0 < avg_resp <= 24:
        score += 0.5
    if signals.get("search_appearance_30d", 0) >= 50:
        score += 0.5

    # Tiered company boost: best ever
    best_tier = 0
    for job in career:
        company = job.get("company", "").strip().lower()
        if is_tier_company(company, TIER_1_COMPANIES):
            best_tier = max(best_tier, 3)
        elif is_tier_company(company, TIER_2_COMPANIES):
            best_tier = max(best_tier, 2)
        elif is_tier_company(company, TIER_3_COMPANIES):
            best_tier = max(best_tier, 1)
    if best_tier == 3:
        score += 1.5
    elif best_tier == 2:
        score += 1.0
    elif best_tier == 1:
        score += 0.5

    # Current company bonus
    curr_company = profile.get("current_company", "").strip().lower()
    if is_tier_company(curr_company, TIER_1_COMPANIES):
        score += 1.0
    elif is_tier_company(curr_company, TIER_2_COMPANIES):
        score += 0.5
    elif is_tier_company(curr_company, TIER_3_COMPANIES):
        score += 0.25

    # Tiered unbroken product streak
    score += tiered_streak_reward(career)

    # NLP/IR depth boost
    nlp_ir_total = sum(
        s.get("duration_months", 0) for s in skills if s["name"].lower() in NLP_IR_SKILLS
    )
    if nlp_ir_total >= 72:
        score += 3.0
    elif nlp_ir_total >= 48:
        score += 1.5

    # Saved by recruiters >= 10 (+1.0)
    if signals.get("saved_by_recruiters_30d", 0) >= 10:
        score += 1.0

    return score

# ---------------------------------------------------------------------------
# PHASE 3: REASONING GENERATION
# ---------------------------------------------------------------------------
def generate_reasoning(c: Dict, rank: int) -> str:
    profile = c["profile"]
    career = c.get("career_history", [])
    signals = c["redrob_signals"]
    skills = c.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    notice = signals.get("notice_period_days", 0)

    sorted_skills = sorted(skills, key=lambda s: s.get("duration_months", 0), reverse=True)
    nlp_skills = [s for s in sorted_skills if s["name"].lower() in NLP_IR_SKILLS]
    top_skills = [s["name"] for s in (nlp_skills[:2] if nlp_skills else sorted_skills[:2])]

    if rank <= 10:
        hook = f"{yoe}yr {current_title} at {current_company} ({industry})"
    elif rank <= 50:
        top_skill = top_skills[0] if top_skills else "ML"
        hook = f"{yoe}yr experience with {top_skill} expertise"
    else:
        hook = f"{yoe}yr {current_title} at {current_company}"

    qual = compute_qualifying_months(c)
    if top_skills:
        evidence = f"Strong in {top_skills[0]}"
        if len(top_skills) > 1:
            evidence += f" and {top_skills[1]}"
    else:
        evidence = "Relevant ML/AI experience"
    evidence += f" ({qual}mo product ML)"

    concern = ""
    if notice > 60:
        concern = f"; {notice}-day notice period is a concern"
    elif signals.get("github_activity_score", -1) == -1:
        concern = "; no GitHub profile linked"

    return f"{hook}. {evidence}{concern}."

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(candidates_file: str, output_file: str) -> None:
    now = datetime(2026, 6, 14, tzinfo=timezone.utc)  # fixed ref date per dataset

    # pass 1: stream and hard-drop
    candidates: List[Dict] = []
    open_func = gzip.open if candidates_file.endswith(".gz") else open
    with open_func(candidates_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if hard_drop(c, now):
                continue
            candidates.append(c)

    # pass 2: TF-IDF similarity over survivors
    if candidates:
        texts = [build_candidate_text(c) for c in candidates] + [JD_TEXT]
        vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
        jd_vec = tfidf[-1]
        cand_vecs = tfidf[:-1]
        similarities = cosine_similarity(cand_vecs, jd_vec).ravel()
    else:
        similarities = []

    survivors = []
    for c, sim in zip(candidates, similarities):
        honeypot = is_honeypot(c)
        qual_months = compute_qualifying_months(c)
        boost = compute_boost_score(c, qual_months, jd_similarity=float(sim))
        penalty = penalty_multiplier(c, now)
        if honeypot:
            penalty *= 0.01
        final_score = boost * penalty
        survivors.append((c["candidate_id"], final_score, c))

    # sort by score desc, candidate_id asc (tie-break); round to match CSV precision
    survivors.sort(key=lambda x: (-round(x[1], 4), x[0]))
    top100 = survivors[:100]

    with open(output_file, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, scr, cand) in enumerate(top100, start=1):
            reasoning = generate_reasoning(cand, i)
            writer.writerow([cid, i, f"{scr:.4f}", reasoning])

    print(f"Done. Survivors: {len(survivors)} | Top 100 written to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--candidates" or sys.argv[3] != "--out":
        print("Usage: python rank.py --candidates <file.jsonl[.gz]> --out <output.csv>")
        sys.exit(1)
    main(sys.argv[2], sys.argv[4])
