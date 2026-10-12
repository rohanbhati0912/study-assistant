import yaml
import streamlit as st
from study_agents import StudyAgents
from rag_helper import RAGHelper
from typing import Optional, Dict, Any

class StudyAssistantHandler:
    def __init__(self, topic, subject_category, knowledge_level, learning_goal, 
                 time_available, learning_style, model_name="gpt-4o", provider="openai"):
        """
        Initialize the study assistant handler.
        
        Args:
            topic (str): The specific topic or subject to study
            subject_category (str): The category of the subject
            knowledge_level (str): Student's current knowledge level
            learning_goal (str): What the student wants to achieve
            time_available (str): How much time the student has
            learning_style (str): Student's preferred learning style
            model_name (str): The model to use
            provider (str): The AI provider ("openai" or "groq")
        """
        self.topic = topic
        self.subject_category = subject_category
        self.knowledge_level = knowledge_level
        self.learning_goal = learning_goal
        self.time_available = time_available
        self.learning_style = learning_style
        self.model_name = model_name
        self.provider = provider
        self.agents = StudyAgents(
            topic, subject_category, knowledge_level, learning_goal,
            time_available, learning_style, model_name, provider
        )
        self.config = self._load_config()
        self.rag_helper = None
    
    def _load_config(self):
        """
        Load the complete configuration from the YAML file.
        
        Returns:
            dict: A dictionary of configuration data
        """
        with open("prompts.yaml", "r") as file:
            return yaml.safe_load(file)
    
    def _format_prompt(self, prompt_template, **kwargs):
        """
        Format a prompt template with variables.
        
        Args:
            prompt_template (str): The prompt template to format
            **kwargs: The variables to insert into the template
            
        Returns:
            str: The formatted prompt
        """
        return prompt_template.format(**kwargs)

    @staticmethod
    def _is_configuration_error(error: ValueError) -> bool:
        """Return whether a ValueError is one raised by StudyAgents configuration."""
        return str(error) in {
            "Unsupported AI provider. Choose Groq or OpenAI.",
            "GROQ_API_KEY is not configured. Add it to your environment and restart the app.",
            "OPENAI_API_KEY is not configured. Add it to your environment and restart the app.",
        }

    @staticmethod
    def _show_provider_error(error: Exception) -> None:
        """Show an actionable provider error without exposing provider response data."""
        error_name = type(error).__name__.lower()
        status_code = getattr(error, "status_code", None)

        if status_code in (401, 403) or "authentication" in error_name or "permission" in error_name:
            st.error("The AI provider rejected your credentials. Check the configured API key and try again.")
        elif status_code == 429 or "ratelimit" in error_name or "rate_limit" in error_name:
            st.error("The AI provider is rate limiting requests. Wait a moment and try again.")
        elif status_code == 404 or "notfound" in error_name or "not_found" in error_name:
            st.error("The selected AI model is unavailable. Choose a supported model and try again.")
        elif any(term in error_name for term in ("connection", "timeout", "connecterror")):
            st.error("The AI provider could not be reached. Check your internet connection and try again.")
        else:
            st.error(
                "The AI request failed. Check your API key, model, and internet "
                "connection, then try again."
            )

    def _run_agent(self, agent_or_factory, prompt):
        """Run an agent and show a safe, actionable error if it fails.

        Passing an agent factory keeps model-configuration errors at the same
        request-handling boundary as provider errors.
        """
        try:
            agent = (
                agent_or_factory
                if hasattr(agent_or_factory, "run")
                else agent_or_factory()
            )
            response = agent.run(prompt, stream=False)
            if response is None or not getattr(response, "content", None):
                st.error("The AI provider returned an empty response. Please retry.")
                return None
            return response.content
        except ValueError as error:
            if self._is_configuration_error(error):
                st.error(str(error))
            else:
                self._show_provider_error(error)
            return None
        except Exception as error:
            self._show_provider_error(error)
            return None
    
    def analyze_student(self):
        """
        Analyze the student's learning needs and create a profile.
        
        Returns:
            dict: Analysis results
        """
        results = {}
        
        with st.status("Analyzing your learning needs...", expanded=True) as status:
            status.update(label="Creating student profile...", state="running")
            
            analysis_prompt = self._format_prompt(
                self.config["prompts"]["student_analysis"]["base"],
                topic=self.topic,
                subject_category=self.subject_category,
                knowledge_level=self.knowledge_level,
                learning_goal=self.learning_goal,
                time_available=self.time_available,
                learning_style=self.learning_style
            )
            
            analysis_result = self._run_agent(
                self.agents.student_analyzer_agent, analysis_prompt
            )
            if analysis_result is None:
                status.update(label="Analysis could not be completed. Please retry.", state="error")
                return {}

            results["analysis"] = analysis_result
            st.session_state.student_analysis = analysis_result
            
            status.update(label="Analysis complete!", state="complete")
        
        return results
    
    def create_roadmap(self, student_analysis: str):
        """
        Create a personalized learning roadmap based on student analysis.
        
        Args:
            student_analysis (str): The student analysis from analyze_student()
            
        Returns:
            dict: Roadmap results
        """
        results = {}
        
        with st.status("Creating your personalized learning roadmap...", expanded=True) as status:
            status.update(label="Designing learning path...", state="running")
            
            roadmap_prompt = self._format_prompt(
                self.config["prompts"]["roadmap_creation"]["base"],
                student_analysis=student_analysis,
                topic=self.topic,
                learning_goal=self.learning_goal,
                time_available=self.time_available,
                knowledge_level=self.knowledge_level
            )
            
            roadmap_result = self._run_agent(
                self.agents.roadmap_creator_agent, roadmap_prompt
            )
            if roadmap_result is None:
                status.update(label="Roadmap could not be created. Please retry.", state="error")
                return {}

            results["roadmap"] = roadmap_result
            st.session_state.learning_roadmap = roadmap_result
            
            status.update(label="Roadmap created!", state="complete")
        
        return results
    
    def find_resources(self):
        """
        Find and recommend learning resources for the topic.
        
        Returns:
            dict: Resource recommendations
        """
        results = {}
        
        with st.status("Finding learning resources...", expanded=True) as status:
            status.update(label="Searching for resources...", state="running")
            
            resource_prompt = self._format_prompt(
                self.config["prompts"]["resource_finding"]["base"],
                topic=self.topic,
                learning_goal=self.learning_goal,
                knowledge_level=self.knowledge_level,
                learning_style=self.learning_style
            )
            
            resource_result = self._run_agent(
                self.agents.resource_finder_agent, resource_prompt
            )
            if resource_result is None:
                status.update(label="Resources could not be found. Please retry.", state="error")
                return {}

            results["resources"] = resource_result
            st.session_state.learning_resources = resource_result
            
            status.update(label="Resources found!", state="complete")
        
        return results
    
    def generate_quiz(self, difficulty_level: str = "intermediate", 
                     focus_areas: str = "general", num_questions: int = 10):
        """
        Generate a quiz to test understanding.
        
        Args:
            difficulty_level (str): The difficulty level of the quiz
            focus_areas (str): Specific areas to focus on
            num_questions (int): Number of questions to generate
            
        Returns:
            dict: Quiz content
        """
        results = {}
        
        with st.status("Generating quiz...", expanded=True) as status:
            status.update(label="Creating questions...", state="running")
            
            quiz_prompt = self._format_prompt(
                self.config["prompts"]["quiz_generation"]["base"],
                topic=self.topic,
                difficulty_level=difficulty_level,
                focus_areas=focus_areas,
                num_questions=num_questions
            )
            
            quiz_result = self._run_agent(
                self.agents.quiz_generator_agent, quiz_prompt
            )
            if quiz_result is None:
                status.update(label="Quiz could not be generated. Please retry.", state="error")
                return {}

            results["quiz"] = quiz_result
            
            status.update(label="Quiz ready!", state="complete")
        
        return results
    
    def get_tutoring(self, student_question: str, context: str = ""):
        """
        Get tutoring help on a specific question.
        
        Args:
            student_question (str): The student's question
            context (str): Additional context
            
        Returns:
            str: Tutoring response
        """
        tutor_prompt = self._format_prompt(
            self.config["prompts"]["tutoring"]["base"],
            student_question=student_question,
            context=context,
            knowledge_level=self.knowledge_level
        )
        
        return self._run_agent(self.agents.tutor_agent, tutor_prompt)
    
    def initialize_rag(self, collection_name: str = "study_materials"):
        """
        Initialize RAG helper for document-based learning.
        
        Args:
            collection_name (str): Name for the document collection
        """
        self.rag_helper = RAGHelper(collection_name=collection_name)
    
    def add_document_to_rag(
        self,
        file_path: str,
        file_type: str = "pdf",
        source_name: Optional[str] = None,
    ) -> bool:
        """
        Add a document to the RAG knowledge base.
        
        Args:
            file_path (str): Path to the document
            file_type (str): Type of document ("pdf" or "text")
            
        Returns:
            bool: Success status
        """
        if not self.rag_helper:
            self.initialize_rag()
        
        if file_type == "pdf":
            return self.rag_helper.load_pdf(file_path, source_name=source_name)
        elif file_type == "text":
            return self.rag_helper.load_text(file_path, source_name=source_name)
        return False
    
    def query_documents(self, question: str, k: int = 4):
        """Query uploaded documents using RAG, including source citations."""
        if not self.rag_helper:
            return "No documents have been uploaded yet. Please upload study materials first."

        # Retrieve relevant chunks with source metadata.
        relevant_docs = self.rag_helper.query_with_sources(question, k=k)

        if not relevant_docs:
            return (
                "I couldn't find relevant information in your uploaded documents. "
                "Please try rephrasing your question or upload more materials."
            )

        # Include source details in the context passed to the tutor.
        context_parts = []
        for index, doc in enumerate(relevant_docs, start=1):
            source = doc.get("source")
            page = doc.get("page")
            citation_parts = []

            if source:
                citation_parts.append(f"Source: {source}")
            if page is not None:
                try:
                    # PyPDFLoader uses zero-based page numbers.
                    page_number = int(page) + 1
                    citation_parts.append(f"Page: {page_number}")
                except (TypeError, ValueError):
                    citation_parts.append(f"Page: {page}")

            citation = f"[{'; '.join(citation_parts)}]" if citation_parts else (
                f"[Retrieved passage {index}; source metadata unavailable]"
            )
            context_parts.append(f"{citation}\n{doc.get('content', '')}")

        context = "\n\n".join(context_parts)

        rag_prompt = self._format_prompt(
            self.config["prompts"]["rag_query"]["base"],
            question=question,
            context=context
        )

        return self._run_agent(self.agents.rag_tutor_agent, rag_prompt)
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the RAG knowledge base.
        
        Returns:
            int: Number of documents
        """
        if not self.rag_helper:
            return 0
        return self.rag_helper.get_document_count()
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from the RAG knowledge base.
        
        Returns:
            bool: Success status
        """
        if not self.rag_helper:
            return False
        return self.rag_helper.clear_database()
