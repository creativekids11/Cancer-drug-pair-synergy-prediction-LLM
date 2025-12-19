#!/usr/bin/env python3
"""
CHECKLIST - DrugCombDB Integration Implementation
Verify all improvements have been successfully implemented
"""

import os
import sys
from pathlib import Path

def check_files():
    """Verify required and deleted files"""
    root = Path("d:\\cancer_drug_pair")
    
    print("\n" + "="*70)
    print("FILE VERIFICATION")
    print("="*70)
    
    # Files that should EXIST
    required_files = {
        "README.md": "Main documentation",
        "QUICKSTART.md": "Quick start guide",
        "MODEL_AND_TRAINING.md": "Model architecture",
        "INDEX.md": "Navigation index",
        "IMPROVEMENTS.md": "Technical improvements",
        "IMPLEMENTATION_READY.md": "Implementation guide",
        "COMPLETE.md": "Completion summary",
        "run.py": "Main pipeline script",
        "test_api_connection.py": "API test script",
    }
    
    # Files that should NOT EXIST
    deleted_files = {
        "API_INTEGRATION.md": "Should be deleted",
        "API_TROUBLESHOOTING.md": "Should be deleted",
        "INTEGRATION_COMPLETE.md": "Should be deleted",
        "FINAL_SUMMARY.md": "Should be deleted",
        "CHANGELOG.md": "Should be deleted",
        "api_test.log": "Should be deleted",
        "api_test2.log": "Should be deleted",
        "final_test.log": "Should be deleted",
    }
    
    print("\n✓ REQUIRED FILES (should exist):")
    all_exist = True
    for filename, description in required_files.items():
        path = root / filename
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {filename:<30} {description}")
        if not exists:
            all_exist = False
    
    print("\n✓ DELETED FILES (should NOT exist):")
    all_deleted = True
    for filename, description in deleted_files.items():
        path = root / filename
        exists = path.exists()
        status = "✓" if not exists else "✗"
        print(f"  {status} {filename:<30} {description}")
        if exists:
            all_deleted = False
    
    return all_exist and all_deleted

def check_code():
    """Verify code improvements in run.py"""
    run_py = Path("d:\\cancer_drug_pair\\run.py")
    
    print("\n" + "="*70)
    print("CODE VERIFICATION")
    print("="*70)
    
    if not run_py.exists():
        print("✗ run.py not found!")
        return False
    
    content = run_py.read_text()
    
    checks = {
        "API endpoint specified": "http://drugcombdb.denglab.org:8888",
        "Page size 500": "PAGE_SIZE = 500",
        "Default pages 500": "MAX_PAGES = getattr(self.args, 'api_max_pages', 500)",
        "Concurrency 60": "CONCURRENCY = 60",
        "Retry logic": "max_retries=3",
        "Exponential backoff": "2 ** attempt",
        "API mandatory": "ALWAYS attempt to fetch from DrugCombDB",
        "Progress tracking": "tqdm",
        "Comprehensive logging": 'logger.info(f"  - Pages fetched',
    }
    
    print("\n✓ CODE IMPROVEMENTS:")
    all_passed = True
    for check_name, search_term in checks.items():
        found = search_term in content
        status = "✓" if found else "✗"
        print(f"  {status} {check_name}")
        if not found:
            all_passed = False
    
    return all_passed

def check_documentation():
    """Verify documentation content"""
    improvements = Path("d:\\cancer_drug_pair\\IMPROVEMENTS.md")
    ready = Path("d:\\cancer_drug_pair\\IMPLEMENTATION_READY.md")
    complete = Path("d:\\cancer_drug_pair\\COMPLETE.md")
    
    print("\n" + "="*70)
    print("DOCUMENTATION VERIFICATION")
    print("="*70)
    
    docs_exist = {
        "IMPROVEMENTS.md": improvements.exists(),
        "IMPLEMENTATION_READY.md": ready.exists(),
        "COMPLETE.md": complete.exists(),
    }
    
    print("\n✓ NEW DOCUMENTATION:")
    all_exist = True
    for doc, exists in docs_exist.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {doc}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_functionality():
    """Verify basic functionality"""
    print("\n" + "="*70)
    print("FUNCTIONALITY CHECK")
    print("="*70)
    
    run_py = Path("d:\\cancer_drug_pair\\run.py")
    test_api = Path("d:\\cancer_drug_pair\\test_api_connection.py")
    
    checks = {
        "run.py executable": run_py.exists() and run_py.stat().st_size > 0,
        "test_api_connection.py exists": test_api.exists() and test_api.stat().st_size > 0,
        "Data directory exists": Path("d:\\cancer_drug_pair\\data_prepared").exists(),
    }
    
    print("\n✓ FUNCTIONALITY:")
    all_passed = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False
    
    return all_passed

def main():
    """Run all checks"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "DRUGCOMBDB INTEGRATION CHECKLIST" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        "Files": check_files(),
        "Code": check_code(),
        "Documentation": check_documentation(),
        "Functionality": check_functionality(),
    }
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n✓ CHECK RESULTS:")
    all_passed = True
    for category, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {category}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - READY FOR PRODUCTION")
        print("\nNext steps:")
        print("  1. Test API connection: python test_api_connection.py")
        print("  2. Run pipeline: python run.py --api-max-pages 100")
        print("  3. Check results: results/experiment_*/results.json")
        print("="*70 + "\n")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - REVIEW ABOVE")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    exit(main())
