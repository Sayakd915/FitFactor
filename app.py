import streamlit as st
import pdfplumber
from resume_screening import analyze_resumes

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using pdfplumber."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def main():
    st.title("Resume Comparison Tool")
    st.markdown("Upload two resume PDFs and enter a job description to see which resume better matches the role.")

    col1, col2 = st.columns(2)
    with col1:
        resume_a_file = st.file_uploader("Upload Resume A (PDF)", type="pdf")
    with col2:
        resume_b_file = st.file_uploader("Upload Resume B (PDF)", type="pdf")

    job_description = st.text_area("Enter Job Description", height=200)

    if st.button("Analyze"):
        if resume_a_file and resume_b_file and job_description:
            resume_a_text = extract_text_from_pdf(resume_a_file)
            resume_b_text = extract_text_from_pdf(resume_b_file)

            if not resume_a_text or not resume_b_text:
                st.error("Failed to extract text from one or both resumes. Please ensure they are text-based PDFs.")
                return

            results = analyze_resumes(job_description, resume_a_text, resume_b_text)

            st.subheader("Results")
            st.write(f"**Resume A Overall Score:** {results['score_a']:.4f}")
            st.write(f"**Resume B Overall Score:** {results['score_b']:.4f}")
            st.write(f"**Better Suited Resume:** {results['better_resume']}")

            st.subheader("Detailed Analysis")
            st.write("The analysis below shows similarity scores for each resume section compared to the job description. Higher scores indicate better alignment.")
            st.table({
                "Section": ["Experience", "Education", "Skills"],
                "Resume A": [results['analysis_data']['Resume A']['experience'],
                             results['analysis_data']['Resume A']['education'],
                             results['analysis_data']['Resume A']['skills']],
                "Resume B": [results['analysis_data']['Resume B']['experience'],
                             results['analysis_data']['Resume B']['education'],
                             results['analysis_data']['Resume B']['skills']]
            })

            st.write("**Why the better resume was chosen:**")
            for line in results['explanation']:
                st.write(line)
        else:
            st.warning("Please upload both resumes and enter a job description to proceed.")

if __name__ == "__main__":
    main()