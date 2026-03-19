import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Course Recommendation System", layout="wide")


@st.cache_data
def load_data():
    data = [
        {
            "title": "Python for Data Science",
            "description": "Learn Python basics for data analysis, data cleaning, and visualization using pandas and numpy.",
            "skills": "python data analysis pandas numpy data science",
            "level": "Beginner",
            "category": "data"
        },
        {
            "title": "Machine Learning Foundations",
            "description": "Introduction to supervised learning, unsupervised learning, model evaluation, and ML workflows.",
            "skills": "machine learning python statistics supervised learning unsupervised learning ml",
            "level": "Intermediate",
            "category": "ml"
        },
        {
            "title": "Deep Learning Fundamentals",
            "description": "Learn neural networks, backpropagation, deep learning architectures, and real-world DL applications.",
            "skills": "deep learning neural networks python ai ml",
            "level": "Advanced",
            "category": "ml"
        },
        {
            "title": "SQL for Analytics",
            "description": "Use SQL to query data, analyze datasets, and solve business intelligence problems.",
            "skills": "sql data analysis database business intelligence analytics",
            "level": "Beginner",
            "category": "data"
        },
        {
            "title": "Natural Language Processing Basics",
            "description": "Understand text preprocessing, embeddings, NLP pipelines, and language model fundamentals.",
            "skills": "nlp python machine learning text processing embeddings ml",
            "level": "Intermediate",
            "category": "ml"
        },
        {
            "title": "Data Visualization with Tableau",
            "description": "Create dashboards and visual reports to communicate insights effectively.",
            "skills": "tableau data visualization dashboards analytics reporting",
            "level": "Beginner",
            "category": "data"
        },
        {
            "title": "Statistics for Machine Learning",
            "description": "Build strong foundations in probability, distributions, and statistics for machine learning.",
            "skills": "statistics probability machine learning math ml",
            "level": "Intermediate",
            "category": "ml"
        },
        {
            "title": "Power BI for Business Intelligence",
            "description": "Build interactive dashboards and business reports using Power BI.",
            "skills": "power bi business intelligence data visualization reporting analytics",
            "level": "Beginner",
            "category": "data"
        },
        {
            "title": "Computer Vision Essentials",
            "description": "Learn image classification, CNNs, and computer vision fundamentals using deep learning.",
            "skills": "computer vision deep learning python cnn ai ml",
            "level": "Advanced",
            "category": "ml"
        },
        {
            "title": "Product Management for AI",
            "description": "Learn AI product strategy, roadmap thinking, metrics, experimentation, stakeholder alignment, and tradeoffs.",
            "skills": "product management ai metrics strategy roadmap experimentation stakeholders",
            "level": "Intermediate",
            "category": "pm"
        },
        {
            "title": "Generative AI for Product Managers",
            "description": "Understand LLMs, prompt design, evaluation, AI risks, and how to build generative AI products.",
            "skills": "generative ai llms prompting ai product management evaluation",
            "level": "Intermediate",
            "category": "pm"
        },
        {
            "title": "User Research and Product Discovery",
            "description": "Learn customer interviews, problem discovery, user journeys, and product opportunity identification.",
            "skills": "user research product discovery customer interviews ux product management",
            "level": "Beginner",
            "category": "pm"
        },
        {
            "title": "MLOps and Model Deployment",
            "description": "Learn model deployment, model monitoring, experiment tracking, CI/CD for ML, and production ML systems.",
            "skills": "mlops model deployment model monitoring machine learning python production ml cicd",
            "level": "Advanced",
            "category": "ml"
        },
        {
            "title": "Feature Engineering for Machine Learning",
            "description": "Learn feature creation, preprocessing, encoding, and data preparation for machine learning models.",
            "skills": "feature engineering preprocessing machine learning data preparation ml python",
            "level": "Intermediate",
            "category": "ml"
        },
        {
            "title": "Data Structures and Algorithms for AI Engineers",
            "description": "Master core data structures and algorithms useful for AI engineering and technical interviews.",
            "skills": "data structures algorithms python problem solving engineering",
            "level": "Intermediate",
            "category": "engineering"
        },
        {
            "title": "Backend Development with APIs",
            "description": "Build REST APIs, work with servers, databases, and backend application logic.",
            "skills": "backend development apis rest api python java databases server",
            "level": "Intermediate",
            "category": "backend"
        },
        {
            "title": "System Design Fundamentals",
            "description": "Learn scalability, databases, caching, load balancing, and distributed systems basics.",
            "skills": "system design scalability databases caching distributed systems load balancing",
            "level": "Advanced",
            "category": "backend"
        },
        {
            "title": "Frontend Development with React",
            "description": "Build interactive web interfaces using HTML, CSS, JavaScript, and React.",
            "skills": "frontend react html css javascript ui web development",
            "level": "Intermediate",
            "category": "frontend"
        },
        {
            "title": "UI UX Design Essentials",
            "description": "Learn user experience principles, wireframing, prototyping, and usability testing.",
            "skills": "ui ux wireframing prototyping design usability figma",
            "level": "Beginner",
            "category": "design"
        },
        {
            "title": "Cloud Computing with AWS",
            "description": "Learn cloud infrastructure, AWS services, deployment, and scalable architecture.",
            "skills": "cloud aws deployment infrastructure scalability devops",
            "level": "Intermediate",
            "category": "cloud"
        },
        {
            "title": "DevOps and CI CD Pipelines",
            "description": "Understand automation, CI CD, Docker, deployment workflows, and DevOps practices.",
            "skills": "devops ci cd docker automation deployment kubernetes",
            "level": "Intermediate",
            "category": "cloud"
        },
        {
            "title": "Cybersecurity Fundamentals",
            "description": "Learn security basics, network security, authentication, vulnerabilities, and risk management.",
            "skills": "cybersecurity network security authentication vulnerabilities risk management",
            "level": "Beginner",
            "category": "security"
        },
        {
            "title": "Data Analyst Career Path",
            "description": "Learn SQL, Excel, dashboards, analytics, and business reporting for data analyst roles.",
            "skills": "data analyst sql excel dashboards analytics reporting power bi tableau",
            "level": "Beginner",
            "category": "data"
        },
        {
            "title": "Excel for Business Analytics",
            "description": "Analyze business data using Excel formulas, pivot tables, charts, and reporting.",
            "skills": "excel analytics reporting pivot tables charts business analysis",
            "level": "Beginner",
            "category": "data"
        }
    ]
    return pd.DataFrame(data)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


ROLE_SKILL_MAP = {
    "ml engineer": {
        "skills": ["python", "machine learning", "statistics", "deep learning", "mlops", "feature engineering"],
        "categories": ["ml", "engineering"],
        "level_pref": ["Intermediate", "Advanced"]
    },
    "data scientist": {
        "skills": ["python", "machine learning", "statistics", "data science", "feature engineering", "nlp"],
        "categories": ["ml", "data"],
        "level_pref": ["Intermediate", "Advanced"]
    },
    "data analyst": {
        "skills": ["sql", "excel", "power bi", "tableau", "analytics", "reporting", "dashboards"],
        "categories": ["data"],
        "level_pref": ["Beginner", "Intermediate"]
    },
    "business analyst": {
        "skills": ["excel", "sql", "analytics", "reporting", "business intelligence", "dashboards"],
        "categories": ["data"],
        "level_pref": ["Beginner", "Intermediate"]
    },
    "ai product manager": {
        "skills": ["product management", "ai", "metrics", "strategy", "experimentation", "user research"],
        "categories": ["pm", "ml"],
        "level_pref": ["Intermediate"]
    },
    "product manager": {
        "skills": ["product management", "strategy", "metrics", "user research", "roadmap"],
        "categories": ["pm"],
        "level_pref": ["Beginner", "Intermediate"]
    },
    "backend engineer": {
        "skills": ["apis", "backend", "databases", "system design", "server", "scalability"],
        "categories": ["backend", "engineering"],
        "level_pref": ["Intermediate", "Advanced"]
    },
    "frontend engineer": {
        "skills": ["react", "html", "css", "javascript", "frontend", "ui"],
        "categories": ["frontend"],
        "level_pref": ["Beginner", "Intermediate"]
    },
    "ui ux designer": {
        "skills": ["ui", "ux", "wireframing", "prototyping", "figma", "usability"],
        "categories": ["design"],
        "level_pref": ["Beginner", "Intermediate"]
    },
    "cloud engineer": {
        "skills": ["aws", "cloud", "deployment", "devops", "docker", "kubernetes"],
        "categories": ["cloud"],
        "level_pref": ["Intermediate", "Advanced"]
    },
    "devops engineer": {
        "skills": ["devops", "docker", "ci cd", "deployment", "automation", "kubernetes"],
        "categories": ["cloud"],
        "level_pref": ["Intermediate", "Advanced"]
    },
    "cybersecurity analyst": {
        "skills": ["cybersecurity", "network security", "authentication", "risk management", "security"],
        "categories": ["security"],
        "level_pref": ["Beginner", "Intermediate"]
    }
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_user_profile(interests: str, skills: str, goal: str) -> str:
    parts = [interests, skills, goal]
    return normalize_text(" ".join([p for p in parts if p and p.strip()]))


def prepare_course_text(df: pd.DataFrame) -> list:
    return (
        df["title"].fillna("") + ". " +
        df["description"].fillna("") + ". " +
        df["skills"].fillna("") + ". " +
        df["level"].fillna("") + ". " +
        df["category"].fillna("")
    ).tolist()


def detect_role(goal: str) -> str | None:
    goal = normalize_text(goal)
    for role in ROLE_SKILL_MAP.keys():
        if role in goal:
            return role
    return None


def skill_overlap_score(course_skills: str, target_skills: list[str]) -> float:
    course_text = normalize_text(course_skills)
    score = 0.0
    for skill in target_skills:
        if skill in course_text:
            score += 0.07
    return min(score, 0.35)


def category_score(course_category: str, target_categories: list[str]) -> float:
    if course_category in target_categories:
        return 0.18
    return -0.08


def level_score(course_level: str, preferred_levels: list[str]) -> float:
    if course_level in preferred_levels:
        return 0.08
    return 0.0


def general_keyword_bonus(user_profile: str, course_text: str) -> float:
    bonus_terms = [
        "python", "machine learning", "statistics", "deep learning", "sql",
        "excel", "power bi", "tableau", "react", "aws", "docker",
        "product management", "user research", "apis", "system design"
    ]
    score = 0.0
    for term in bonus_terms:
        if term in user_profile and term in course_text:
            score += 0.03
    return min(score, 0.15)


def generate_recommendations(user_profile: str, goal: str, df: pd.DataFrame, model, top_k: int = 5) -> pd.DataFrame:
    course_texts = prepare_course_text(df)

    course_embeddings = model.encode(course_texts)
    user_embedding = model.encode([user_profile])
    semantic_scores = cosine_similarity(user_embedding, course_embeddings)[0]

    detected_role = detect_role(goal)
    final_scores = []

    for i, row in df.iterrows():
        score = float(semantic_scores[i])
        course_text = normalize_text(course_texts[i])

        score += general_keyword_bonus(user_profile, course_text)

        if detected_role:
            role_info = ROLE_SKILL_MAP[detected_role]
            score += skill_overlap_score(row["skills"], role_info["skills"])
            score += category_score(row["category"], role_info["categories"])
            score += level_score(row["level"], role_info["level_pref"])

        final_scores.append(score)

    results = df.copy()
    results["score"] = final_scores
    results = results.sort_values(by="score", ascending=False).head(top_k)
    return results


def get_reason(course_row: pd.Series, goal: str) -> str:
    detected_role = detect_role(goal)

    if detected_role:
        role_info = ROLE_SKILL_MAP[detected_role]
        matched_skills = [
            skill for skill in role_info["skills"]
            if skill in normalize_text(course_row["skills"] + " " + course_row["description"])
        ]
        if matched_skills:
            return f"Relevant for {detected_role} because it matches: {', '.join(matched_skills[:3])}"

    return "Recommended based on semantic similarity to your interests and career goal."


st.title("AI-Powered Course Recommendation System")
st.markdown("Get personalized course recommendations based on your interests, skills, and career goals.")

df = load_data()
model = load_model()

with st.sidebar:
    st.header("Your Profile")
    interests = st.text_input("Interests", placeholder="e.g. machine learning, AI")
    skills = st.text_input("Existing Skills", placeholder="e.g. python, statistics")
    goal = st.text_input("Career Goal", placeholder="e.g. become an ML engineer")
    top_k = st.slider("Number of Recommendations", min_value=3, max_value=8, value=5)

if st.button("Recommend Courses"):
    user_profile = build_user_profile(interests, skills, goal)

    if not user_profile.strip():
        st.warning("Please enter at least one of: interests, skills, or career goal.")
    else:
        recommendations = generate_recommendations(user_profile, goal, df, model, top_k=top_k)

        st.subheader("Top Recommended Courses")

        for _, row in recommendations.iterrows():
            reason = get_reason(row, goal)

            with st.container():
                st.markdown(f"### {row['title']}")
                st.write(f"**Level:** {row['level']}")
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Skills:** {row['skills']}")
                st.write(f"**Recommendation Score:** {row['score']:.3f}")
                st.write(f"**Why recommended:** {reason}")
                st.markdown("---")

st.subheader("Course Catalog")
st.dataframe(df[["title", "level", "category", "skills"]], use_container_width=True)
