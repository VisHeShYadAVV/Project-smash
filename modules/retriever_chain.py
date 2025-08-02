# modules/retriever_chain.py

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from modules.prompts import prompt_template

async def get_answers_as_json(questions, vector_store):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY", "")
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser()
    answers = []

    for question in questions:
        docs_and_scores = vector_store.similarity_search_with_score(question, k=5)
        filtered_docs = [doc for doc, score in docs_and_scores if score > 0.75]

        if not filtered_docs:
            answers.append("The provided document does not contain information about this topic.")
            continue

        input_data = {
            "context": "\n\n".join(doc.page_content for doc in filtered_docs),
            "questions": question
        }

        try:
            chain = prompt | llm | parser
            result = chain.invoke(input_data)
            answer = result.get("answers", [""])[0]
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error answering question: {str(e)}")

    return {"answers": answers}
