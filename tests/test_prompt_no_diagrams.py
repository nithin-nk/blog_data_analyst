"""
Tests to verify that mermaid diagram and visual diagram references
have been removed from all prompts (content generator, outline generator, outline reviewer).

These tests ensure that diagram generation is not mentioned in prompts,
as diagrams will be handled by a separate agent.
"""

import pytest
import inspect
from src.generation.content_generator import ContentGenerator
from src.planning.outline_generator import OutlineGenerator
from src.planning.outline_reviewer import OutlineReviewer


class TestContentGeneratorPrompts:
    """Test that ContentGenerator prompts don't reference diagrams."""
    
    def test_generate_section_content_initial_prompt_no_mermaid(self):
        """Verify initial generation prompt doesn't mention Mermaid."""
        generator = ContentGenerator()
        
        # Inspect the method source code to verify prompt content
        source = inspect.getsource(generator._generate_section_content)
        
        # Verify "Mermaid" is not in the prompt
        assert "Mermaid" not in source, "Initial prompt should not mention Mermaid diagrams"
        # Check case-insensitive for "use mermaid"
        lowercase_source = source.lower()
        assert "use mermaid" not in lowercase_source, \
            "Initial prompt should not mention using mermaid"
        
    def test_generate_section_content_prompt_has_code_only(self):
        """Verify prompt mentions CODE but not DIAGRAMS."""
        generator = ContentGenerator()
        
        source = inspect.getsource(generator._generate_section_content)
        
        # Should have CODE section
        assert "**CODE**:" in source, "Prompt should have CODE section"
        
        # Should NOT have DIAGRAMS or CODE & DIAGRAMS
        assert "CODE & DIAGRAMS" not in source, "Prompt should not have 'CODE & DIAGRAMS'"
        assert "CODE &amp; DIAGRAMS" not in source, "Prompt should not have 'CODE &amp; DIAGRAMS'"
        
    def test_generate_section_content_prompt_no_diagram_instructions(self):
        """Verify prompt doesn't instruct to use diagrams."""
        generator = ContentGenerator()
        
        source = inspect.getsource(generator._generate_section_content)
        
        # Check for common diagram-related phrases
        diagram_phrases = [
            "Use Mermaid diagrams",
            "Include diagrams",
            "visual diagrams",
            "architecture diagrams",
            "flow diagrams",
            "diagram for"
        ]
        
        for phrase in diagram_phrases:
            assert phrase not in source, f"Prompt should not contain '{phrase}'"


class TestOutlineGeneratorPrompts:
    """Test that OutlineGenerator prompts don't reference diagrams."""
    
    def test_system_prompt_no_mermaid_references(self):
        """Verify SYSTEM_PROMPT doesn't mention Mermaid."""
        prompt = OutlineGenerator.SYSTEM_PROMPT
        
        assert "Mermaid" not in prompt, "Outline generator should not mention Mermaid"
        assert "mermaid" not in prompt.lower(), "Outline generator should not mention mermaid (case insensitive)"
        
    def test_system_prompt_no_diagram_references(self):
        """Verify SYSTEM_PROMPT doesn't mention diagrams."""
        prompt = OutlineGenerator.SYSTEM_PROMPT
        
        # Check for various diagram references
        diagram_phrases = [
            "visual diagrams",
            "architecture diagrams", 
            "Include Mermaid diagram",
            "diagram showing",
            "flow diagram",
            "sequence diagram"
        ]
        
        for phrase in diagram_phrases:
            assert phrase not in prompt, f"Outline generator should not contain '{phrase}'"
            
    def test_system_prompt_has_code_specifications_only(self):
        """Verify prompt has CODE SPECIFICATIONS but not CODE & DIAGRAM SPECIFICATIONS."""
        prompt = OutlineGenerator.SYSTEM_PROMPT
        
        # Should have CODE SPECIFICATIONS
        assert "CODE SPECIFICATIONS:" in prompt, "Should have CODE SPECIFICATIONS section"
        
        # Should NOT have CODE & DIAGRAM SPECIFICATIONS
        assert "CODE & DIAGRAM SPECIFICATIONS:" not in prompt, \
            "Should not have 'CODE & DIAGRAM SPECIFICATIONS'"
        assert "CODE &amp; DIAGRAM SPECIFICATIONS:" not in prompt, \
            "Should not have 'CODE &amp; DIAGRAM SPECIFICATIONS' (HTML entity)"
            
    def test_system_prompt_content_philosophy_no_visual_diagrams(self):
        """Verify CONTENT PHILOSOPHY doesn't mention visual diagrams."""
        prompt = OutlineGenerator.SYSTEM_PROMPT
        
        # Find the CONTENT PHILOSOPHY section
        assert "CONTENT PHILOSOPHY:" in prompt
        
        # It should say "Mix conceptual explanations with hands-on code examples"
        # NOT "Mix conceptual explanations with hands-on code examples and visual diagrams"
        assert "hands-on code examples and visual diagrams" not in prompt, \
            "Content philosophy should not mention 'visual diagrams'"
        assert "hands-on code examples" in prompt, \
            "Content philosophy should mention hands-on code examples"


class TestOutlineReviewerPrompts:
    """Test that OutlineReviewer prompts don't reference diagrams."""
    
    def test_system_prompt_no_diagram_in_evaluation_criteria(self):
        """Verify evaluation criteria don't mention diagrams."""
        prompt = OutlineReviewer.SYSTEM_PROMPT
        
        # Should not ask about diagram instructions
        assert "code/diagram instructions" not in prompt, \
            "Reviewer should not evaluate 'code/diagram instructions'"
        
        # Should ask about code instructions only
        assert "code instructions" in prompt, \
            "Reviewer should evaluate code instructions"
            
    def test_system_prompt_balance_criteria_no_diagrams(self):
        """Verify Balance & Content Mix criteria doesn't mention diagrams."""
        prompt = OutlineReviewer.SYSTEM_PROMPT
        
        # Should not mention "code examples and diagrams"
        assert "code examples and diagrams" not in prompt, \
            "Balance criteria should not mention diagrams"
        
        # Should mention code examples
        assert "code examples" in prompt, \
            "Balance criteria should mention code examples"
            
    def test_system_prompt_no_mermaid_references(self):
        """Verify reviewer prompt has no Mermaid references."""
        prompt = OutlineReviewer.SYSTEM_PROMPT
        
        assert "Mermaid" not in prompt, "Reviewer should not mention Mermaid"
        assert "mermaid" not in prompt.lower(), "Reviewer should not mention mermaid (case insensitive)"


class TestIntegrationNoMermaidInWorkflow:
    """Integration tests to ensure no mermaid references in the workflow."""
    
    def test_all_prompts_combined_no_mermaid(self):
        """Verify no Mermaid references across all prompts."""
        content_gen_source = ContentGenerator._generate_section_content.__code__.co_consts
        outline_prompt = OutlineGenerator.SYSTEM_PROMPT
        reviewer_prompt = OutlineReviewer.SYSTEM_PROMPT
        
        # Check all prompts
        all_text = str(content_gen_source) + outline_prompt + reviewer_prompt
        
        # Case-insensitive check for mermaid (excluding variable names)
        assert all_text.lower().count("mermaid") == 0, \
            "No prompts should contain 'mermaid' references"
    
    def test_code_mentioned_diagrams_not_mentioned(self):
        """Verify code is mentioned but diagrams are not."""
        outline_prompt = OutlineGenerator.SYSTEM_PROMPT
        reviewer_prompt = OutlineReviewer.SYSTEM_PROMPT
        
        # Code should be mentioned
        assert "code" in outline_prompt.lower(), "Outline should mention code"
        assert "code" in reviewer_prompt.lower(), "Reviewer should mention code"
        
        # Diagrams should not be mentioned (except in context like "code examples and diagrams" which we removed)
        # We'll check that the specific harmful phrases are gone
        harmful_phrases = [
            "and diagrams",
            "or diagrams", 
            "& diagrams",
            "and visual diagrams",
            "Include Mermaid"
        ]
        
        for phrase in harmful_phrases:
            assert phrase not in outline_prompt, \
                f"Outline should not contain '{phrase}'"
            assert phrase not in reviewer_prompt, \
                f"Reviewer should not contain '{phrase}'"
