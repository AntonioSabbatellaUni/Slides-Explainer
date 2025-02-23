"""
Context Manager Module
--------------------
Handles hierarchical context management and user constraints for slide explanations.
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slide_explainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default constraint templates
DEFAULT_CONSTRAINTS = {
    'style': {
        'formal': 'Maintain formal academic language',
        'technical': 'Focus on technical details and specifications',
        'simple': 'Use simple, accessible language',
        'detailed': 'Provide comprehensive explanations'
    },
    'content': {
        'definitions': 'Include definitions for technical terms',
        'examples': 'Provide practical examples',
        'comparisons': 'Include comparisons with related concepts',
        'applications': 'Highlight practical applications'
    },
    'structure': {
        'bullet_points': 'Structure explanation with bullet points',
        'narrative': 'Present as a flowing narrative',
        'step_by_step': 'Break down into sequential steps'
    }
}

class ContextManager:
    """Manages hierarchical context and constraints for slide explanations."""
    
    def __init__(self, context_size: int = 3):
        """
        Initialize the context manager.
        
        Parameters
        ----------
        context_size : int
            Maximum number of slides to keep in local context.
        """
        self.context_size = context_size
        self.local_context = []
        self.global_context = {
            'summary': '',
            'key_themes': [],
            'last_update': None
        }
        self.context_counter = 0
        self.active_constraints = set()
        self.user_constraints = {}
        
        logger.info(f"Initialized ContextManager with context_size={context_size}")
    
    def add_explanation(
        self,
        explanation: str,
        slide_index: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a new explanation to the context.
        
        Parameters
        ----------
        explanation : str
            The slide explanation to add.
        slide_index : int
            Index of the slide.
        metadata : Optional[Dict]
            Additional metadata about the explanation.
        """
        timestamp = datetime.now()
        
        # Create explanation entry
        entry = {
            'index': slide_index,
            'explanation': explanation,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Add to local context
        self.local_context.append(entry)
        self.context_counter += 1
        
        logger.info(f"Added explanation for slide {slide_index} to local context")
        
        # Check if we need to summarize
        if len(self.local_context) > self.context_size:
            self._summarize_and_rotate()
    
    def _summarize_and_rotate(self) -> None:
        """Summarize current local context and rotate it."""
        try:
            # Keep the last 1-2 slides in local context
            to_summarize = self.local_context[:-2]
            self.local_context = self.local_context[-2:]
            
            logger.info(f"Starting summarization of {len(to_summarize)} slides")
            logger.info(f"Slide indices to summarize: {[e['index'] for e in to_summarize]}")
            
            # Update global context with summary
            old_summary = self.global_context['summary']
            summary = self._generate_summary(to_summarize)
            self.global_context['summary'] = summary
            self.global_context['last_update'] = datetime.now()
            
            logger.info("Summary rotation completed:")
            logger.info(f"- Old summary length: {len(old_summary) if old_summary else 0} chars")
            logger.info(f"- New summary length: {len(summary)} chars")
            logger.info(f"- Keeping slides {[e['index'] for e in self.local_context]} in local context")
            
        except Exception as e:
            logger.error(f"Error in summarize_and_rotate: {e}")
            logger.error("Stack trace:", exc_info=True)
            # Keep the most recent context in case of error
            self.local_context = self.local_context[-self.context_size:]
    
    def _generate_summary(self, explanations: List[Dict]) -> str:
        """
        Generate a summary from a list of explanations.
        
        Parameters
        ----------
        explanations : List[Dict]
            List of explanation entries to summarize.
            
        Returns
        -------
        str
            Generated summary.
        """
        try:
            logger.info(f"Generating summary for {len(explanations)} explanations")
            
            # TODO: Implement actual summarization logic
            # For now, just concatenate with markers
            summary_parts = []
            total_chars = 0
            
            for e in explanations:
                excerpt = e['explanation'][:100] + "..."
                summary_parts.append(f"Slide {e['index']}: {excerpt}")
                total_chars += len(e['explanation'])
            
            summary = "\n".join(summary_parts)
            
            logger.info("Summary generation completed:")
            logger.info(f"- Input: {len(explanations)} explanations, {total_chars} total chars")
            logger.info(f"- Output: {len(summary)} chars in summary")
            logger.info(f"- Compression ratio: {len(summary)/total_chars:.2%}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            logger.error("Stack trace:", exc_info=True)
            return "Error generating summary"
    
    def build_prompt(self, current_slides: str) -> str:
        """
        Build a structured prompt including context and constraints.
        
        Parameters
        ----------
        current_slides : str
            Description or content of current slides.
            
        Returns
        -------
        str
            Structured prompt for the model.
        """
        try:
            logger.info("Building structured prompt")
            
            # Build prompt sections
            goal = "Explain the current slides in the context of the presentation flow"
            expected_output = "Detailed explanation connecting current content with previous context"
            
            # Add active constraints
            constraints = "\n".join([
                f"- {self.user_constraints[c]}"
                for c in self.active_constraints
            ]) if self.active_constraints else "None"
            
            # Add context
            global_context = self.global_context['summary'] if self.global_context['summary'] else "None"
            local_context = "\n".join([
                f"Slide {e['index']}: {e['explanation']}"
                for e in self.local_context
            ]) if self.local_context else "None"
            
            # Construct final prompt
            prompt = f"""
Goal: {goal}

Expected Output: {expected_output}

User Constraints:
{constraints}

Context:
1. Global Summary:
{global_context}

2. Recent Slides:
{local_context}

3. Current Slides:
{current_slides}
"""
            
            logger.info("Prompt built with:")
            logger.info(f"- Active constraints: {len(self.active_constraints)}")
            logger.info(f"- Global context: {'Present' if global_context != 'None' else 'None'}")
            logger.info(f"- Local context: {len(self.local_context)} slides")
            logger.info(f"- Total prompt length: {len(prompt)} chars")
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            logger.error("Stack trace:", exc_info=True)
            # Return a basic prompt in case of error
            return f"Please explain these slides: {current_slides}"
    
    def add_constraint(self, constraint_id: str, constraint_text: str) -> None:
        """
        Add a new user constraint.
        
        Parameters
        ----------
        constraint_id : str
            Unique identifier for the constraint.
        constraint_text : str
            The constraint text.
        """
        self.user_constraints[constraint_id] = constraint_text
        logger.info(f"Added new constraint: [{constraint_id}] {constraint_text[:50]}...")
    
    def _log_constraint_changes(self, action: str, constraints: list) -> None:
        """
        Log multiple constraint changes in a single message.
        
        Parameters
        ----------
        action : str
            The action performed ('activated' or 'deactivated').
        constraints : list
            List of constraint IDs that were changed.
        """
        if constraints:
            logger.info(f"Constraints {action}: {constraints}")
    
    def update_constraints(self, active_constraints: Set[str]) -> None:
        """
        Update the active constraints based on the new set.
        
        Parameters
        ----------
        active_constraints : Set[str]
            Set of constraint IDs that should be active.
        """
        currently_active = self.active_constraints.copy()
        new_active = set(active_constraints)
        
        # Find constraints to activate and deactivate
        to_activate = new_active - currently_active
        to_deactivate = currently_active - new_active
        
        # Update the active constraints
        self.active_constraints = new_active
        
        # Log changes in batches
        if to_activate:
            self._log_constraint_changes("activated", list(to_activate))
        if to_deactivate:
            self._log_constraint_changes("deactivated", list(to_deactivate))
    
    def activate_constraint(self, constraint_id: str) -> None:
        """
        Activate a constraint.
        
        Parameters
        ----------
        constraint_id : str
            ID of the constraint to activate.
        """
        if constraint_id in self.user_constraints and constraint_id not in self.active_constraints:
            self.active_constraints.add(constraint_id)
            self._log_constraint_changes("activated", [constraint_id])
    
    def deactivate_constraint(self, constraint_id: str) -> None:
        """
        Deactivate a constraint.
        
        Parameters
        ----------
        constraint_id : str
            ID of the constraint to deactivate.
        """
        if constraint_id in self.active_constraints:
            self.active_constraints.discard(constraint_id)
            self._log_constraint_changes("deactivated", [constraint_id])
    
    def clear_context(self) -> None:
        """Clear all context."""
        self.local_context = []
        self.global_context = {
            'summary': '',
            'key_themes': [],
            'last_update': None
        }
        self.context_counter = 0
        if self.active_constraints:
            self._log_constraint_changes("cleared", list(self.active_constraints))
            self.active_constraints.clear()
        logger.info("Cleared all context") 