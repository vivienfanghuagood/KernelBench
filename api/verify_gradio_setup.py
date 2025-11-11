#!/usr/bin/env python3
"""
Verification script for Gradio frontend implementation.
Run this to check if all files are in place and dependencies can be imported.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed - run: pip install {module_name}")
        return False

def main():
    print("=" * 70)
    print("KernelBench Gradio Frontend Verification")
    print("=" * 70)
    
    print("\n📁 Checking Files...")
    files_ok = all([
        check_file_exists("api/gradio_app.py", "Main Gradio app"),
        check_file_exists("api/start_gradio.py", "Standalone launcher"),
        check_file_exists("api/requirements.txt", "Requirements file"),
        check_file_exists("api/main.py", "FastAPI main"),
        check_file_exists("api/README.md", "API README"),
        check_file_exists("api/GRADIO_SETUP.md", "Gradio setup guide"),
        check_file_exists("api/GRADIO_IMPLEMENTATION.md", "Implementation summary"),
    ])
    
    print("\n📦 Checking Dependencies...")
    deps_ok = all([
        check_import("gradio"),
        check_import("requests"),
        check_import("fastapi"),
        check_import("uvicorn"),
    ])
    
    print("\n🔍 Checking API Backend...")
    try:
        import requests
        response = requests.get("http://localhost:8009/health", timeout=2)
        if response.ok:
            print("✅ FastAPI backend is running at http://localhost:8009")
            backend_ok = True
        else:
            print("⚠️  FastAPI backend responded but not healthy")
            backend_ok = False
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI backend is NOT running at http://localhost:8009")
        print("   Start it with: cd api && python main.py")
        backend_ok = False
    except ImportError:
        print("⚠️  Cannot check backend (requests not installed)")
        backend_ok = False
    except Exception as e:
        print(f"⚠️  Error checking backend: {e}")
        backend_ok = False
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if files_ok and deps_ok:
        print("✅ All files present and dependencies installed!")
        print("\n📝 Next Steps:")
        print("   1. Start FastAPI backend: cd api && python main.py")
        print("   2. Access Gradio at: http://localhost:8009/gradio")
        print("   3. Or run standalone: cd api && python start_gradio.py")
        return 0
    elif files_ok and not deps_ok:
        print("⚠️  Files OK but dependencies missing")
        print("\n📝 Install dependencies:")
        print("   cd api && pip install -r requirements.txt")
        return 1
    else:
        print("❌ Some files are missing or dependencies not installed")
        print("\n📝 Please check the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
