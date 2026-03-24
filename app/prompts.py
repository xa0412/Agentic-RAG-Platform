from langchain_core.prompts import ChatPromptTemplate

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful research assistant. Use the provided context documents to \
answer the question when they are relevant. If the context is not relevant or the \
question is general knowledge (e.g. math, science, definitions), answer from your \
own knowledge. Always be accurate and concise.

Context documents:
{context}""",
    ),
    ("human", "{question}"),
])

ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a grader checking if an answer properly addresses a question.
Respond with ONLY "yes" if the answer is relevant and complete, or "no" if it is \
not helpful, vague, or says it lacks information.""",
    ),
    (
        "human",
        "Question: {question}\n\nAnswer: {answer}\n\nDoes this answer address the question?",
    ),
])
