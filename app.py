
from conversation_memory import reformulate_question
from reranker import rerank_documents
from hybrid_search import hybrid_search
import streamlit as st
import hashlib

from document_processor import process_multiple_pdfs
from rag_pipeline import generate_answer


# ==========================================
# 1. Page configuration
# ==========================================
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered"
)


# ==========================================
# 2. Application title
# ==========================================
st.title("📚 RAG Document Assistant")
st.caption("Upload multiple PDFs and ask questions across their content.")


# ==========================================
# 3. Initialize session state
# ==========================================
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ==========================================
# 4. Citation display function
# ==========================================
def display_sources(sources):

    with st.expander("📚 Sources & Evidence", expanded=False):

        for i, source in enumerate(sources, start=1):

            st.markdown(f"**Source {i}**")

            st.write(
                f"📄 {source.get('file_name', 'Uploaded PDF')} "
                f"— Page {source.get('page', 'Unknown')}"
            )

            excerpt = source.get("excerpt", "")

            if excerpt:
                st.markdown(f"> {excerpt}")

            st.divider()


# ==========================================
# 5. Sidebar - Multiple PDF upload
# ==========================================
uploaded_files = None
file_hash = None

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Select one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        # Generate a combined fingerprint for all PDFs
        hash_object = hashlib.sha256()

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()

            hash_object.update(uploaded_file.name.encode("utf-8"))
            hash_object.update(b"\0")
            hash_object.update(hashlib.sha256(file_bytes).digest())

        file_hash = hash_object.hexdigest()

        st.write("**Selected documents:**")

        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

        # Process all selected PDFs
        if st.button(
            "🔍 Process All Documents",
            use_container_width=True
        ):

            with st.spinner(
                "Reading, chunking and indexing all PDFs..."
            ):

                try:

                    vector_store, pages, chunk_count, chunks = (
                        process_multiple_pdfs(uploaded_files)
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.page_count = pages
                    st.session_state.chunk_count = chunk_count

                    st.session_state.processed_file_hash = file_hash

                    st.session_state.processed_file_names = [
                        file.name for file in uploaded_files
                    ]

                    st.session_state.processed_file_name = ", ".join(
                        st.session_state.processed_file_names
                    )

                    # Reset conversation for the new document collection
                    st.session_state.messages = []

                    st.success("All documents processed successfully!")

                except Exception as e:

                    st.error(f"Processing failed: {e}")

        # Display processing status
        if st.session_state.processed_file_hash == file_hash:

            st.success("✅ Documents ready")

            st.metric(
                "PDF files",
                len(st.session_state.processed_file_names)
            )

            st.metric(
                "Total pages",
                st.session_state.page_count
            )

            st.metric(
                "Total chunks",
                st.session_state.chunk_count
            )

        else:

            st.info("Click 'Process All Documents' to begin.")

    else:

        st.info("Upload one or more PDF files to get started.")

    st.divider()

    st.header("ℹ️ About")

    st.write(
        "This assistant searches across multiple uploaded PDFs, "
        "retrieves relevant information, and generates answers using AI."
    )

    st.caption("Powered by Python, FAISS, LangChain & Ollama")

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()


# ==========================================
# 6. Check document readiness
# ==========================================
document_ready = (
    bool(uploaded_files)
    and st.session_state.vector_store is not None
    and st.session_state.processed_file_hash == file_hash
)


if not document_ready:

    st.info(
        "👈 Upload your PDFs in the sidebar and click "
        "**Process All Documents** to start chatting."
    )


# ==========================================
# 7. Display chat history
# ==========================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:
                display_sources(sources)


# ==========================================
# 8. Chat input
# ==========================================
question = st.chat_input(
    "Ask a question across your PDFs...",
    disabled=not document_ready
)


# ==========================================
# 9. Process user question
# ==========================================
if question and document_ready:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                # Build conversation history, excluding the latest question
                chat_history = ""

                for message in st.session_state.messages[:-1]:
                    chat_history += (
                        f"{message['role']}: {message['content']}\n"
                    )

                # Convert follow-up question into a standalone question
                standalone_question = reformulate_question(
                    question,
                    chat_history
                )

                print("Original question:", question)
                print("Standalone question:", standalone_question)

                # Step 1: Hybrid search across all uploaded PDF chunks
                candidate_documents = hybrid_search(
                    standalone_question,
                    st.session_state.chunks,
                    st.session_state.vector_store,
                    k=10
                )

                # Step 2: Rerank candidate documents
                documents = rerank_documents(
                    standalone_question,
                    candidate_documents,
                    top_n=3
                )

                # Step 3: Generate answer
                if not documents:

                    answer = (
                        "I couldn't find this information "
                        "in the provided documents."
                    )

                    citations = []

                else:

                    context = "\n\n".join(
                        doc.page_content for doc in documents
                    )

                    answer = generate_answer(
                        standalone_question,
                        context
                    )

                    # Step 4: Prepare citations
                    citations = []

                    for doc in documents:

                        metadata = doc.metadata

                        page_number = metadata.get("page", 0) + 1

                        file_name = metadata.get(
                            "source",
                            "Uploaded PDF"
                        )

                        citations.append({
                            "file_name": file_name,
                            "page": page_number,
                            "excerpt": doc.page_content.strip()
                        })

                # Display answer
                st.markdown(answer)

                # Display citations
                if citations:
                    display_sources(citations)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": citations
                })

            except Exception as e:

                st.error(f"An error occurred: {e}")
