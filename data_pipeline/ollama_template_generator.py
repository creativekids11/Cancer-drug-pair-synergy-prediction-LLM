#!/usr/bin/env python3
"""
ollama_template_generator.py

Dedicated script for generating 100+ prompt templates using Ollama 3.1
Can be run standalone or integrated with main pipeline
"""

import json
import logging
import argparse
import time
from typing import List, Dict, Optional
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("ollama_gen")

class OllamaTemplateGenerator:
    """Advanced template generation using Ollama 3.1"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2:latest",
        timeout: int = 120
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.generated_count = 0
        self.failed_count = 0
    
    def health_check(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                log.info(f"✓ Ollama is running with {len(models)} models")
                log.info(f"  Target model: {self.model}")
                return True
            else:
                log.error(f"✗ Ollama returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            log.error(f"✗ Cannot connect to Ollama at {self.base_url}")
            log.error("  Make sure Ollama is installed and running:")
            log.error("  $ ollama serve")
            return False
        except Exception as e:
            log.error(f"✗ Health check failed: {e}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(2, 4, 30))
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API with retry logic"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": 200,
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                if result:
                    return result
                return None
            else:
                log.warning(f"API returned {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            log.warning("Request timeout - retrying...")
            raise
        except Exception as e:
            log.error(f"API call failed: {e}")
            return None
    
    def generate_single_template(self, template_num: int) -> Optional[Dict]:
        """Generate a single template"""
        system_instructions = """You are an expert in oncology and drug interaction pharmacology.
Your task is to generate a UNIQUE prompt template for predicting drug synergy in cancer cells.

REQUIREMENTS:
1. The template must include placeholders for: {drugA}, {drugB}, {cell_line}, {tissue}
2. Ask for a clear Yes/No classification
3. Vary the structure and phrasing significantly from other templates
4. Include scientific context about why synergy prediction matters
5. Be 3-6 sentences long

INSTRUCTIONS:
- Start with different question types: predictive, analytical, investigative, diagnostic
- Vary the order of information presentation
- Include domain context: mechanism of action, cancer biology, pharmacology
- Use professional but accessible language
- End with a clear Yes/No question

OUTPUT ONLY the template text without any explanation or numbering."""

        user_prompt = f"""Generate a unique drug synergy prediction template #{template_num}.
Make it significantly different from standard formats.
Use all four placeholders: {{drugA}}, {{drugB}}, {{cell_line}}, {{tissue}}."""

        full_prompt = f"{system_instructions}\n\n{user_prompt}"
        
        response = self._call_ollama(full_prompt)
        
        if response and len(response) > 30:
            # Clean and validate response
            response = response.strip()
            
            # Ensure placeholders are in correct format
            response = response.replace("{{drugA}}", "{drugA}")
            response = response.replace("{{drugB}}", "{drugB}")
            response = response.replace("{{cell_line}}", "{cell_line}")
            response = response.replace("{{tissue}}", "{tissue}")
            
            # Validate all fields are present
            required_fields = ["drugA", "drugB", "cell_line", "tissue"]
            if all(f"{{{f}}}" in response for f in required_fields):
                return {
                    "template": response,
                    "fields": required_fields,
                    "source": "ollama",
                    "model": self.model,
                    "template_number": template_num,
                    "length": len(response)
                }
            else:
                log.debug(f"Template #{template_num} missing required fields")
                return None
        
        return None
    
    def generate_batch(self, n: int = 100, batch_size: int = 5) -> List[Dict]:
        """Generate n templates in batches"""
        log.info(f"Generating {n} templates using {self.model}")
        log.info("=" * 60)
        
        templates = []
        failed = 0
        
        with tqdm(total=n, desc="Templates") as pbar:
            for i in range(1, n + 1):
                try:
                    template = self.generate_single_template(i)
                    
                    if template:
                        templates.append(template)
                        pbar.update(1)
                    else:
                        failed += 1
                    
                    # Small delay between requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    log.error(f"Template #{i} generation failed: {e}")
                    failed += 1
        
        log.info("=" * 60)
        log.info(f"Generated: {len(templates)} | Failed: {failed}")
        
        self.generated_count = len(templates)
        self.failed_count = failed
        
        return templates
    
    def save_templates(self, templates: List[Dict], output_path: str) -> None:
        """Save templates to JSONL file"""
        with open(output_path, "w") as f:
            for t in templates:
                f.write(json.dumps(t) + "\n")
        
        log.info(f"✓ Saved {len(templates)} templates to {output_path}")
    
    def generate_report(self, templates: List[Dict]) -> Dict:
        """Generate statistics about generated templates"""
        lengths = [t.get("length", 0) for t in templates]
        
        return {
            "total": len(templates),
            "model": self.model,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "generation_time": time.time(),
        }

def main():
    parser = argparse.ArgumentParser(
        description="Generate 100+ prompt templates using Ollama 3.1"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of templates to generate (default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama2:latest",
        help="Ollama model to use (e.g., llama2, neural-chat, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ollama_templates.jsonl",
        help="Output file path (default: ollama_templates.jsonl)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics without generating"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = OllamaTemplateGenerator(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout
    )
    
    # Check health
    if not generator.health_check():
        log.error("Cannot proceed without Ollama running")
        return 1
    
    # Generate templates
    log.info(f"Generating {args.count} templates...")
    templates = generator.generate_batch(n=args.count)
    
    # Save results
    if templates:
        generator.save_templates(templates, args.output)
        
        # Print statistics
        report = generator.generate_report(templates)
        log.info("\nGeneration Report:")
        log.info(f"  Total generated: {report['total']}")
        log.info(f"  Avg length: {report['avg_length']:.0f} chars")
        log.info(f"  Min/Max: {report['min_length']}-{report['max_length']} chars")
        log.info(f"  Model: {report['model']}")
        
        return 0
    else:
        log.error("No templates generated")
        return 1

if __name__ == "__main__":
    exit(main())
