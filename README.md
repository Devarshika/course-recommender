# AI-Powered Course Recommendation System

An AI-assisted course recommendation system that suggests relevant learning paths based on a user’s interests, existing skills, and career goals.

This project was built as an AI Product Management portfolio project to demonstrate how recommendation systems can improve course discovery, personalize learning experiences, and support better user outcomes.

---

## Problem Statement

Online learning platforms offer thousands of courses, but many users struggle to decide what to learn next. Generic recommendations often fail to reflect a learner’s real goals, current skill level, or intended career path.

This leads to:
- poor course discovery
- low engagement
- irrelevant enrollments
- lower completion rates

The core problem is helping users discover the right courses at the right time.

---

## Solution

This system uses a hybrid recommendation approach to generate personalized course suggestions.

It combines:
- semantic similarity using sentence embeddings
- role-based skill mapping
- category matching
- level preference matching

The application takes user inputs such as:
- interests
- existing skills
- career goal

It then recommends the most relevant courses and explains why they were selected.

---

## Product Thinking Behind the Project

This project was designed as more than a coding demo. It also reflects product management thinking around personalization systems.

Key product considerations included:
- **Cold start problem:** users may have no prior learning history, so the app uses onboarding-style inputs instead
- **Exploration vs exploitation:** recommendations should balance user intent with adjacent relevant learning opportunities
- **Career-goal alignment:** recommendations should reflect the user’s target role, not only broad keyword similarity
- **User trust:** the app explains why a course was recommended to improve transparency

---

## How It Works

1. The user enters interests, existing skills, and a career goal
2. The system builds a user profile from these inputs
3. Sentence embeddings are used to measure semantic similarity between the user profile and course descriptions
4. Additional ranking signals are applied:
   - role-to-skill match
   - category alignment
   - level preference
5. The system returns top-ranked courses with recommendation scores and reasons

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Sentence Transformers
- Scikit-learn
- Torch

---

## Project Structure

```bash
ai-course-recommender/
├── app.py
├── requirements.txt
└── README.md

<img width="1897" height="900" alt="Screenshot 2026-03-19 100529" src="https://github.com/user-attachments/assets/59fdb16b-c472-436a-997e-d895016069d0" />
<img width="1910" height="920" alt="Screenshot 2026-03-19 100548" src="https://github.com/user-attachments/assets/175483e8-61b2-4e6b-9938-b77236f196f6" />
<img width="1907" height="906" alt="Screenshot 2026-03-19 100603" src="https://github.com/user-attachments/assets/9a825f22-d56d-4d1d-9daa-2c5ead74fce2" />
