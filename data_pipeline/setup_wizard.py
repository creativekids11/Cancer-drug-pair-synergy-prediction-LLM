#!/usr/bin/env python3
"""
QUICK START GUIDE - CancerGPT Pipeline with Ollama 3.1

This script provides step-by-step setup and execution instructions.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(title: str) -> None:
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def check_python() -> bool:
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_dependencies() -> bool:
    """Check if required packages are installed"""
    required = [
        "pandas", "numpy", "sklearn", "aiohttp", 
        "requests", "tenacity", "tqdm"
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg}")
            missing.append(pkg)
    
    return len(missing) == 0, missing

def install_dependencies(missing: list) -> bool:
    """Install missing dependencies"""
    if not missing:
        return True
    
    print(f"\nInstalling {len(missing)} missing packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing
        )
        return True
    except subprocess.CalledProcessError:
        print("✗ Installation failed")
        return False

def check_ollama() -> tuple:
    """Check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, models
    except:
        pass
    return False, []

def print_ollama_setup() -> None:
    """Print Ollama setup instructions"""
    print_header("OLLAMA SETUP")
    
    print("Ollama enables LLM-generated prompt templates.")
    print("It's optional but recommended for best results.\n")
    
    print("1. Download from: https://ollama.com/download")
    print("   - Windows: ollama-windows.exe")
    print("   - macOS: ollama-macos.zip")
    print("   - Linux: curl https://ollama.ai/install.sh | sh\n")
    
    print("2. Pull a model in terminal:")
    print("   $ ollama pull llama2:latest")
    print("   # or for llama 3.1:")
    print("   $ ollama pull llama3.1:latest\n")
    
    print("3. Start Ollama (keeps running):")
    print("   $ ollama serve\n")
    
    print("4. In another terminal, run the pipeline:")
    print("   $ python orchestrator_tissue_restricted.py run_all\n")

def main():
    """Main setup wizard"""
    os.system("clear" if os.name == "posix" else "cls")
    
    print_header("CANCERGPT PIPELINE - SETUP WIZARD")
    
    # Step 1: Python
    print("STEP 1: Checking Python...")
    if not check_python():
        print("\n✗ Please install Python 3.8 or later")
        return False
    
    # Step 2: Dependencies
    print("\nSTEP 2: Checking Dependencies...")
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\nMissing: {', '.join(missing)}")
        response = input("Install now? [y/n] ").lower()
        if response == "y":
            if not install_dependencies(missing):
                return False
        else:
            print("Skipping installation (you may encounter errors)")
    
    # Step 3: Ollama
    print("\nSTEP 3: Checking Ollama...")
    ollama_available, models = check_ollama()
    
    if ollama_available:
        print(f"✓ Ollama is running with {len(models)} models")
        if models:
            for model in models[:3]:
                print(f"  - {model.get('name', 'unknown')}")
    else:
        print("✗ Ollama not running (optional)")
        response = input("Setup Ollama now? [y/n] ").lower()
        if response == "y":
            print_ollama_setup()
            return True
    
    # Step 4: Ready to run
    print_header("READY TO START")
    
    print("Choose your option:\n")
    print("1. Full pipeline (ingestion → templates → output)")
    print("   $ python orchestrator_tissue_restricted.py run_all\n")
    
    print("2. Generate templates only")
    print("   $ python orchestrator_tissue_restricted.py templates_only\n")
    
    print("3. Ingest data only")
    print("   $ python orchestrator_tissue_restricted.py ingest\n")
    
    print("4. Generate templates with Ollama (standalone)")
    print("   $ python ollama_template_generator.py --count 100\n")
    
    print("5. Run with custom synergy threshold")
    print("   $ python orchestrator_tissue_restricted.py run_all --synergy-threshold 6.5\n")
    
    print("For more options and detailed documentation:")
    print("  → See IMPROVEMENTS.md\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
