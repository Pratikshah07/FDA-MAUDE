"""
Simple script to run the MAUDE processor application.
"""
import os

from app import app

if __name__ == '__main__':
    # Verify API key is set
    if not os.getenv('GROQ_API_KEY'):
        print("=" * 60)
        print("WARNING: GROQ_API_KEY environment variable not set!")
        print("Please set it before running:")
        print("  Windows: set GROQ_API_KEY=your_key_here")
        print("  Linux/Mac: export GROQ_API_KEY=your_key_here")
        print("=" * 60)
        print()
    else:
        print("GROQ_API_KEY configured")
    
    print("Starting MAUDE Data Processor...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
