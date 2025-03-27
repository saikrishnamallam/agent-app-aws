from typing import Optional, List
from textwrap import dedent
import os
from pathlib import Path

from agno.agent import Agent, AgentKnowledge
from agno.models.together import Together
from agno.embedder.together import TogetherEmbedder
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.pdf import PDFKnowledgeBase

from app.core.config import settings
from db.session import db_url


def initialize_legal_knowledge_base(
    documents_path: str = "data/documents",
    table_name: str = "legal_knowledge"
) -> None:
    """
    Initialize the legal knowledge base with documents from the specified path.
    
    Args:
        documents_path: Path to the directory containing legal documents
        table_name: Name of the table to store the knowledge base
    """
    # Create documents directory if it doesn't exist
    os.makedirs(documents_path, exist_ok=True)
    
    # Initialize the knowledge base with Together AI embeddings
    knowledge_base = PDFKnowledgeBase(
        path=documents_path,
        vector_db=PgVector(
            table_name=table_name,
            db_url=db_url,
            search_type=SearchType.hybrid,
            embedder=TogetherEmbedder(
                id=settings.TOGETHER_EMBEDDING_MODEL,
                api_key=settings.TOGETHER_API_KEY
            )
        )
    )
    
    # Load documents into the knowledge base
    knowledge_base.load()
    
    print(f"Legal knowledge base initialized with documents from {documents_path}")


def get_legal_agent(
    model_id: str = settings.TOGETHER_MODEL,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    additional_context = ""
    if user_id:
        additional_context += "<context>"
        additional_context += f"You are interacting with the user: {user_id}"
        additional_context += "</context>"

    return Agent(
        name="Legal Assistant",
        agent_id="legal",
        user_id=user_id,
        session_id=session_id,
        model=Together(
            id=model_id,
            api_key=settings.TOGETHER_API_KEY,
            temperature=0.1,
            top_p=0.9,
            max_tokens=settings.AGENT_MAX_TOKENS
        ),
        # Storage for the agent
        storage=PostgresAgentStorage(table_name="legal_sessions", db_url=db_url),
        # Knowledge base for the agent with Together AI embeddings
        knowledge=AgentKnowledge(
            vector_db=PgVector(
                table_name="legal_knowledge",
                db_url=db_url,
                search_type=SearchType.hybrid,
                embedder=TogetherEmbedder(
                    id=settings.TOGETHER_EMBEDDING_MODEL,
                    api_key=settings.TOGETHER_API_KEY
                )
            )
        ),
        # Description of the agent
        description=dedent("""\
            You are a specialized legal assistant focusing on Italian law and legal documents.
            You have access to a comprehensive knowledge base of legal documents and can provide
            detailed information about legal matters with proper citations and references.
        """),
        # Instructions for the agent
        instructions=dedent("""\
            Respond to legal queries by following these steps:

            1. Knowledge Base Search
            - Analyze the user's legal query and identify key legal terms and concepts
            - Search the knowledge base for relevant legal documents and precedents
            - Always prioritize information from the knowledge base over general knowledge

            2. Response Construction
            - Start with a clear, concise answer addressing the legal query
            - Include specific citations to legal articles, laws, and regulations
            - Format citations as: [Document Title, Article X]
            - Maintain a formal and precise legal tone throughout
            - If information is not found in the knowledge base, clearly state this limitation

            3. Legal Context & References
            - Provide relevant legal context and background information
            - Reference specific sections of laws and regulations
            - Include relevant legal precedents when available
            - Maintain professional and ethical standards in all responses

            4. Quality Assurance
            - Ensure all legal citations are accurate and properly formatted
            - Verify that interpretations align with established legal principles
            - Maintain consistency with previous responses
            - Clearly distinguish between established law and legal interpretations

            5. Engagement & Follow-up
            - Suggest related legal topics for further exploration
            - Ask clarifying questions when the query is ambiguous
            - Offer to provide more detailed information on specific aspects
        """),
        additional_context=additional_context,
        # Format responses using markdown
        markdown=True,
        # Add the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Send the last 3 messages from the chat history
        add_history_to_messages=True,
        num_history_responses=3,
        # Add a tool to read the chat history if needed
        read_chat_history=True,
        # Show debug logs
        debug_mode=debug_mode,
    ) 