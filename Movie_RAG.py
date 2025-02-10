from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import ollama
import json


class NodePoolSystem:
    def __init__(self, db_path):
        self.loader = TextLoader(db_path)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.vectorstore = None

    def initialize(self):
        """Initialize the system"""
        documents = self.loader.load()

        # Split node entries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n---", "\n\nMovie Title:"]
        )

        split_documents = splitter.split_documents(documents)

        # Create vector store
        self.vectorstore = FAISS.from_documents(
            split_documents,
            self.embeddings
        )


class SolutionGenerator:
    def __init__(self, system):
        self.system = system
        self.retriever = self.system.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'score_threshold': 0.6}
        )

    def generate(self, task):
        # Retrieve relevant methods
        docs = self.retriever.get_relevant_documents(task)

        # Build context
        context_blocks = []
        for doc in docs:
            context_blocks.append(doc.page_content)
        context = "\n\n--- separator ---\n\n".join(context_blocks)

        # Design interference-resistant prompt (consider user personality, e.g., user is positive...)
        prompt = f"""You are a movie recommendation expert. Answer based on the following context: \n(Note: Context may contain irrelevant content - identify movie-related information yourself)\n

{context}

Question: {task}
Requirements:
1. Select only movie-related entries
2. Recommend strictly based on the information above
3. Explain reasons for each recommendation
"""

        print(prompt)
        # Call LLM
        response = ollama.chat(
            model='deepseek-r1:7b',
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']


# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    # Initialize system
    system = NodePoolSystem("/home/cxr-ubuntu/Documents/HWU/Movie-Recommender/database/knowledge_base.txt")
    system.initialize()

    # Define task
    task = """
    Could you recommend two movies for me?"
    """

    # Generate solution
    generator = SolutionGenerator(system)
    solution = generator.generate(task)

    print("\n=== Generation Answer ===")
    print(solution)