
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from rag_pipeline import ask_question
from ragas_evaluator import evaluator_llm, evaluator_embeddings


# 1. Prepare test questions
test_questions = [
    "How many sick leave days are employees entitled to?",
    "What is the company's leave policy?",
    "What is the company's office address?"
]


# 2. Prepare evaluation data
questions = []
answers = []
contexts = []

print("\nGenerating answers for evaluation...\n")


# 3. Run questions through our RAG pipeline
for question in test_questions:

    print("Question:", question)

    answer, sources = ask_question(question)

    retrieved_contexts = [
        doc.page_content for doc in sources
    ]

    questions.append(question)
    answers.append(answer)
    contexts.append(retrieved_contexts)

    print("Answer:", answer)
    print("-" * 50)


# 4. Create Ragas dataset
evaluation_dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts
})


# 5. Run Ragas evaluation
print("\nStarting Ragas evaluation...\n")

results = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        faithfulness,
        answer_relevancy
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


# 6. Display evaluation results
print("\n===== RAGAS EVALUATION RESULTS =====")

print(results)

# 7. Save detailed results
df = results.to_pandas()

df.to_csv(
    "ragas_evaluation_results.csv",
    index=False
)

print("\nResults saved to ragas_evaluation_results.csv")
