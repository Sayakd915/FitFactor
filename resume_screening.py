import re
from sentence_transformers import SentenceTransformer, util

def preprocess_text(text):
    """Convert text to lowercase and remove non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def extract_sections(text):
    """Extract experience, education, and skills sections from resume text."""
    sections = {
        'experience': re.findall(r'experience\n(.*?)(?:education|skills|$)', text, re.DOTALL),
        'education': re.findall(r'education\n(.*?)(?:experience|skills|$)', text, re.DOTALL),
        'skills': re.findall(r'skills\n(.*?)(?:experience|education|$)', text, re.DOTALL),
    }
    return {k: v[0].strip() if v else '' for k, v in sections.items()}

def analyze_resumes(job_description, resume_a_text, resume_b_text):
    """Analyze and compare two resumes against a job description."""
    job_description = preprocess_text(job_description)
    resume_a_sections = extract_sections(preprocess_text(resume_a_text))
    resume_b_sections = extract_sections(preprocess_text(resume_b_text))

    # Load pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_embedding = model.encode(job_description)

    # Define section weights
    sections = ['experience', 'education', 'skills']
    weights = {'experience': 0.5, 'skills': 0.3, 'education': 0.2}

    def compute_section_similarity(resume_sections):
        """Compute similarity scores for each resume section."""
        similarities = {}
        for section in sections:
            if resume_sections[section]:
                section_embedding = model.encode(resume_sections[section])
                similarities[section] = util.pytorch_cos_sim(job_embedding, section_embedding).item()
            else:
                similarities[section] = 0.0
        return similarities

    sim_a = compute_section_similarity(resume_a_sections)
    sim_b = compute_section_similarity(resume_b_sections)

    # Calculate weighted overall scores
    def weighted_similarity(sim_dict):
        return sum(sim_dict[section] * weights[section] for section in sections)

    score_a = weighted_similarity(sim_a)
    score_b = weighted_similarity(sim_b)

    # Determine the better resume
    better_resume = "Resume A" if score_a > score_b else "Resume B" if score_b > score_a else "Both are equally suited"

    # Prepare detailed analysis
    analysis_data = {
        'Resume A': {section: f"{sim_a[section]:.4f}" for section in sections},
        'Resume B': {section: f"{sim_b[section]:.4f}" for section in sections}
    }

    # Generate detailed explanation
    explanation = []
    if better_resume == "Resume A":
        explanation.append("- Resume A has a higher overall score, indicating stronger relevance to the job description.")
        if float(analysis_data['Resume A']['experience']) > float(analysis_data['Resume B']['experience']):
            explanation.append("- Stronger match in the Experience section, which carries the most weight (50%).")
        if float(analysis_data['Resume A']['skills']) > float(analysis_data['Resume B']['skills']):
            explanation.append("- Better alignment of Skills (30% weight) with the job requirements.")
        if float(analysis_data['Resume A']['education']) > float(analysis_data['Resume B']['education']):
            explanation.append("- More relevant Education background (20% weight).")
    elif better_resume == "Resume B":
        explanation.append("- Resume B has a higher overall score, indicating stronger relevance to the job description.")
        if float(analysis_data['Resume B']['experience']) > float(analysis_data['Resume A']['experience']):
            explanation.append("- Stronger match in the Experience section, which carries the most weight (50%).")
        if float(analysis_data['Resume B']['skills']) > float(analysis_data['Resume A']['skills']):
            explanation.append("- Better alignment of Skills (30% weight) with the job requirements.")
        if float(analysis_data['Resume B']['education']) > float(analysis_data['Resume A']['education']):
            explanation.append("- More relevant Education background (20% weight).")
    else:
        explanation.append("- Both resumes have equal overall scores, suggesting similar suitability for the role.")

    return {
        "score_a": score_a,
        "score_b": score_b,
        "better_resume": better_resume,
        "analysis_data": analysis_data,
        "explanation": explanation
    }