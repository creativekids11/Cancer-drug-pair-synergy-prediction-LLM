#!/usr/bin/env python3
"""
prompt_utils.py

Advanced prompt utilities for drug synergy prediction:
- Template generation strategies
- Few-shot example builders
- Prompt validation
- Template testing utilities
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

log = logging.getLogger("prompt_utils")

class PromptTemplate:
    """Represents a single prompt template with metadata"""
    
    def __init__(
        self,
        template: str,
        fields: List[str],
        source: str = "static",
        description: str = "",
        model: str = ""
    ):
        self.template = template
        self.fields = fields
        self.source = source
        self.description = description
        self.model = model
    
    def format(self, row: Dict) -> str:
        """Format template with data from a row"""
        try:
            return self.template.format(**{f: row.get(f, "") for f in self.fields})
        except KeyError as e:
            log.warning(f"Missing field in template: {e}")
            return ""
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate template has all required fields"""
        errors = []
        
        if not self.template or len(self.template) < 10:
            errors.append("Template too short")
        
        for field in self.fields:
            if f"{{{field}}}" not in self.template:
                errors.append(f"Field {field} not in template")
        
        if not self.fields:
            errors.append("No fields specified")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "template": self.template,
            "fields": self.fields,
            "source": self.source,
            "description": self.description,
            "model": self.model,
        }

class FewShotBuilder:
    """Build few-shot prompts with examples"""
    
    def __init__(self, examples_df: pd.DataFrame, n_examples: int = 3):
        """
        Initialize with examples
        
        Args:
            examples_df: DataFrame with columns [drugA, drugB, cell_line, tissue, synergy_label]
            n_examples: Number of examples to include
        """
        self.examples_df = examples_df
        self.n_examples = n_examples
    
    def get_balanced_examples(self) -> pd.DataFrame:
        """Get balanced positive/negative examples"""
        synergy = self.examples_df[self.examples_df["synergy_label"] == 1]
        non_synergy = self.examples_df[self.examples_df["synergy_label"] == 0]
        
        n_per_class = self.n_examples // 2
        
        examples = pd.concat([
            synergy.sample(min(n_per_class, len(synergy))),
            non_synergy.sample(min(n_per_class, len(non_synergy)))
        ])
        
        return examples.reset_index(drop=True)
    
    def build_few_shot_prompt(
        self,
        template: PromptTemplate,
        test_row: Dict,
        examples: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Build few-shot prompt with examples
        
        Args:
            template: PromptTemplate to use
            test_row: Row to predict for
            examples: DataFrame of examples (uses balanced if None)
        
        Returns:
            Complete few-shot prompt
        """
        if examples is None:
            examples = self.get_balanced_examples()
        
        prompt = "You are an expert in drug interaction prediction. Learn from these examples:\n\n"
        
        # Add examples
        for idx, row in examples.iterrows():
            example_prompt = template.format(row)
            answer = "Yes" if row["synergy_label"] == 1 else "No"
            prompt += f"Example {idx+1}:\n{example_prompt}\nAnswer: {answer}\n\n"
        
        # Add test case
        prompt += "Now predict for:\n"
        prompt += template.format(test_row)
        prompt += "\nAnswer (Yes/No):"
        
        return prompt

class TemplateValidator:
    """Validate and test prompt templates"""
    
    @staticmethod
    def validate_templates(templates: List[Dict]) -> Tuple[int, List[str]]:
        """
        Validate list of templates
        
        Returns:
            (valid_count, list of errors)
        """
        valid = 0
        errors = []
        
        for i, t in enumerate(templates):
            pt = PromptTemplate(
                template=t.get("template", ""),
                fields=t.get("fields", []),
                source=t.get("source", ""),
            )
            is_valid, errs = pt.validate()
            
            if is_valid:
                valid += 1
            else:
                errors.append(f"Template {i}: {', '.join(errs)}")
        
        return valid, errors
    
    @staticmethod
    def test_template_on_sample(
        template: PromptTemplate,
        sample_row: Dict
    ) -> Tuple[bool, str]:
        """
        Test template on a sample row
        
        Returns:
            (success, formatted_prompt)
        """
        try:
            prompt = template.format(sample_row)
            return len(prompt) > 0, prompt
        except Exception as e:
            return False, str(e)

class PromptTemplateLibrary:
    """Load and manage prompt template library"""
    
    def __init__(self, templates_jsonl: str):
        self.templates = []
        self.load_from_jsonl(templates_jsonl)
    
    def load_from_jsonl(self, path: str) -> None:
        """Load templates from JSONL file"""
        try:
            with open(path, "r") as f:
                for line in f:
                    t = json.loads(line)
                    pt = PromptTemplate(
                        template=t.get("template", ""),
                        fields=t.get("fields", []),
                        source=t.get("source", ""),
                        description=t.get("description", ""),
                        model=t.get("model", ""),
                    )
                    self.templates.append(pt)
            log.info(f"Loaded {len(self.templates)} templates from {path}")
        except FileNotFoundError:
            log.warning(f"Templates file not found: {path}")
        except Exception as e:
            log.error(f"Error loading templates: {e}")
    
    def get_by_source(self, source: str) -> List[PromptTemplate]:
        """Get templates by source (static/ollama)"""
        return [t for t in self.templates if t.source == source]
    
    def sample_templates(self, n: int = 5) -> List[PromptTemplate]:
        """Randomly sample templates"""
        import random
        return random.sample(self.templates, min(n, len(self.templates)))
    
    def validate_all(self) -> Dict:
        """Validate all templates"""
        valid_count, errors = TemplateValidator.validate_templates(
            [t.to_dict() for t in self.templates]
        )
        
        return {
            "total": len(self.templates),
            "valid": valid_count,
            "invalid": len(self.templates) - valid_count,
            "errors": errors,
            "static": len(self.get_by_source("static")),
            "ollama": len(self.get_by_source("ollama")),
        }

def create_metadata_report(templates: List[Dict]) -> Dict:
    """Create metadata report for templates"""
    sources = {}
    field_usage = {}
    
    for t in templates:
        source = t.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        
        for field in t.get("fields", []):
            field_usage[field] = field_usage.get(field, 0) + 1
    
    return {
        "total_templates": len(templates),
        "sources": sources,
        "field_usage": field_usage,
        "avg_template_length": sum(len(t.get("template", "")) for t in templates) / len(templates) if templates else 0,
    }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample template
    template = PromptTemplate(
        template="Drug 1: {drugA}\nDrug 2: {drugB}\nCell: {cell_line}\nTissue: {tissue}\n\nSynergistic?",
        fields=["drugA", "drugB", "cell_line", "tissue"],
        source="static"
    )
    
    # Validate
    is_valid, errors = template.validate()
    print(f"Template valid: {is_valid}")
    
    # Test
    sample = {
        "drugA": "Aspirin",
        "drugB": "Ibuprofen",
        "cell_line": "HeLa",
        "tissue": "breast"
    }
    
    formatted = template.format(sample)
    print(f"\nFormatted prompt:\n{formatted}")
