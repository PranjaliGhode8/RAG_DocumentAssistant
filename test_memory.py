from conversation_memory import reformulate_question

history = """
User: What is the company's annual leave policy?
Assistant: Employees receive 20 days of annual leave per year.
"""

question = "Can I carry those forward?"

result = reformulate_question(question, history)

print("Original question:", question)
print("Reformulated question:", result)