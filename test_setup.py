#!/usr/bin/env python3
"""
Test script to verify the installation and API keys
"""

import os
import sys

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'crewai',
        'crewai_tools',
        'langchain_community',
        'langchain_openai',
        'openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_env_variables():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'SERPER_API_KEY': 'Serper API Key'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value.startswith('your_'):
            print(f"❌ {description} not set or invalid")
            missing_vars.append(var)
        else:
            # Show partial key for verification
            masked_key = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
            print(f"✓ {description} set: {masked_key}")
    
    if missing_vars:
        print("\nPlease set the following in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_actual_key_here")
        return False
    
    return True

def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print("✓ OpenAI API connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("AI Financial Trading Crew - Installation Test")
    print("=" * 60)
    print()
    
    print("Checking Python version...")
    python_ok = check_python_version()
    print()
    
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("Checking environment variables...")
    env_ok = check_env_variables()
    print()
    
    if python_ok and deps_ok and env_ok:
        print("Testing OpenAI connection...")
        api_ok = test_openai_connection()
        print()
        
        if api_ok:
            print("=" * 60)
            print("✅ All checks passed! You're ready to run the application.")
            print("=" * 60)
            print()
            print("To start the application, run:")
            print("  streamlit run app.py")
            print()
            return 0
    
    print("=" * 60)
    print("❌ Some checks failed. Please fix the issues above.")
    print("=" * 60)
    return 1

if __name__ == "__main__":
    exit(main())
