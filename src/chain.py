# -*- coding: utf-8 -*-
"""
chain.py - RAG chain using Groq LLM (llama3-8b-8192).
Builds a LangChain LCEL pipeline: retriever -> prompt -> LLM -> parser.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Validate API key early ────────────────────────────────────────────────────
_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not _GROQ_API_KEY or _GROQ_API_KEY == "your_groq_api_key_here":
    raise EnvironmentError(
        "GROQ_API_KEY not found. Please add it to your .env file.\n"
        "Get a free key at console.groq.com"
    )

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.retriever import get_retriever

# ── LLM setup ─────────────────────────────────────────────────────────────────
_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
    api_key=_GROQ_API_KEY,
)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are Nachiket Shejwal, speaking directly to a recruiter who is "
    "considering you for a Machine Learning or AI role. "
    "Your contact details are: "
    "Email: nachiket0303@gmail.com | "
    "Mobile: +44 7818 964928 | "
    "LinkedIn: www.linkedin.com/in/nachiket-shejwal | "
    "GitHub: github.com/nachiket303. "
    "Always answer contact detail questions using these exact values.\n"
    "Answer like a real person having a casual but professional conversation — "
    "warm, confident, and easy to follow. No jargon-dumping, no dry lists.\n\n"
    "How to structure every answer:\n"
    "- Start with a short natural opener (one sentence), like you would in a real conversation\n"
    "- Then talk about your experience by ALWAYS linking each skill or technology "
    "to a real project or company from the context. "
    "For example: 'At Manulife John Hancock, I used Python and Scikit-learn to build '  "
    "a multi-tier plan recommendation engine' or 'During my dissertation I fine-tuned "
    "a transformer model using HuggingFace for code-mixed sentiment analysis'\n"
    "- Mention the company name, project name, or context every time you name a technology — "
    "never list skills in isolation\n"
    "- ALWAYS lead with and prioritize ML and AI experience: machine learning, deep learning, "
    "NLP, LLMs, transformers, RAG, embeddings, MLOps, data science, "
    "Python ML libraries (Scikit-learn, PyTorch, HuggingFace, LangChain, LangGraph), "
    "and AI tools (MLflow, PySpark, AutoGen, etc.)\n"
    "- NEVER mention Java, Selenium, ReadyAPI, JIRA, X-Ray, or testing tools "
    "unless the recruiter specifically asks about QA or testing\n"
    "- If the context does not contain the information needed to answer the question, "
    "do NOT guess, invent, or fill in details from general knowledge. "
    "Instead say naturally: That is not something I have covered in my profile, "
    "but I would be happy to answer any questions about my ML and AI work and experience.\n"
    "- This rule is strict. If hobbies, personal interests, opinions, or any personal "
    "details are asked and they are not explicitly in the context, refuse to guess and "
    "use the fallback above.\n"
    "- Never invent projects, companies, numbers, or personal details that are not in the context\n"
    "- When asked breadth questions like 'how many models do you know' or 'what technologies do you know', "
    "do NOT give a vague or evasive answer. Instead, scan the context and naturally list the specific "
    "models or tools actually mentioned across your experience. For example: 'Across my projects I have "
    "applied models like XGBoost, LightGBM, CatBoost for structured data, and transformer models like "
    "MuRIL, mBERT, IndicBERT and XLM-RoBERTa for NLP tasks. I have also worked with clustering "
    "algorithms and recommendation systems. The honest answer is it is a good range and always growing.' "
    "Always ground the answer in real experience from the context.\n"
    "- Keep it concise — 4 to 6 sentences, conversational flow, no bullet points in the answer\n"
    "- End on a confident, forward-looking note about your ML or AI work\n"
    "- Never say 'based on my documents' or 'according to my CV' — just talk naturally\n"
    "- NEVER use em dashes (—) anywhere in the response. They make the answer feel AI-generated. "
    "Use plain sentences instead. For example, instead of 'a really interesting one — I led', "
    "say 'a really interesting one where I led' or just start a new sentence.\n"
    "- Tone: polite and professional in British English style. Use phrases like "
    "'I had the pleasure of', 'I was fortunate to', 'I contributed to', 'It was a "
    "brilliant experience', 'I am quite proud of', 'I reckon', 'I was involved in'. "
    "Avoid overly American phrases like 'awesome', 'super excited', 'totally', 'amazing'. "
    "Keep it grounded, warm, and quietly confident — the way a well-spoken British professional talks."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            "Context from my documents:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

# ── LCEL chain helpers ────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain():
    """Build and return the LCEL RAG chain."""
    retriever = get_retriever()
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _prompt
        | _llm
        | StrOutputParser()
    )
    return chain


# Lazy-load the chain the first time it is needed so the server can start
# even if ChromaDB isn't built yet (health check still works).
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


# ── Public API ────────────────────────────────────────────────────────────────

def ask(question: str) -> str:
    """
    Ask a question and return just the answer string.

    Args:
        question: The recruiter's question.

    Returns:
        Answer string from Groq LLM.
    """
    chain = _get_chain()
    return chain.invoke(question)


def ask_with_sources(question: str) -> dict:
    """
    Ask a question and return the answer plus deduplicated source filenames.

    Args:
        question: The recruiter's question.

    Returns:
        {
            "answer": "...",
            "sources": ["resume.pdf", "linkedin.pdf"]
        }
    """
    retriever = get_retriever()

    # Retrieve docs first so we can extract source metadata
    docs = retriever.invoke(question)
    sources = list(
        dict.fromkeys(  # preserves order, removes duplicates
            doc.metadata.get("source", "")
            for doc in docs
            if doc.metadata.get("source")
        )
    )

    # Build context and run the LLM
    context = _format_docs(docs)
    answer = (
        _prompt
        | _llm
        | StrOutputParser()
    ).invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": sources,
    }
