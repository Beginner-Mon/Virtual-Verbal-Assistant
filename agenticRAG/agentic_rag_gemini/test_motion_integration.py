#!/usr/bin/env python3
"""
Quick test script to verify motion generation integration.

This script tests the complete pipeline:
1. Query with visualize_motion intent
2. Exercise name extraction 
3. Motion generation call to DART
4. Response format validation

Usage:
    python test_motion_integration.py
"""

import json
import requests
import time

def test_visualize_motion_integration():
    """Test the complete motion generation pipeline."""
    
    # Test queries that should trigger visualize_motion intent
    test_cases = [
        "Visualize chin tuck",
        "Show me how to do a squat", 
        "Demonstrate shoulder roll",
        "Animate neck stretch"
    ]
    
    base_url = "http://localhost:8080"
    
    print("🧪 Testing Motion Generation Integration")
    print("=" * 50)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n📍 Test {i}: {query}")
        print("-" * 30)
        
        # Prepare request
        payload = {
            "query": query,
            "user_id": "test_user",
            "conversation_history": []
        }
        
        try:
            # Call the orchestrator API
            start_time = time.perf_counter()
            response = requests.post(
                f"{base_url}/answer",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            end_time = time.perf_counter()
            
            data = response.json()
            duration_ms = (end_time - start_time) * 1000
            
            # Analyze response
            print(f"✅ Response received in {duration_ms:.0f}ms")
            print(f"📝 Text answer: {data.get('text_answer', 'N/A')[:100]}...")
            
            # Check motion generation
            motion = data.get('motion')
            if motion:
                print(f"🎬 Motion generated:")
                print(f"   - File: {motion.get('motion_file_url', 'N/A')}")
                print(f"   - Frames: {motion.get('num_frames', 'N/A')}")
                print(f"   - FPS: {motion.get('fps', 'N/A')}")
                print(f"   - Duration: {motion.get('duration_seconds', 'N/A')}s")
                print(f"   - Prompt: {motion.get('text_prompt', 'N/A')}")
            else:
                print("❌ No motion data returned")
            
            # Check exercises
            exercises = data.get('exercises', [])
            if exercises:
                print(f"🏋️ Exercises found: {[e.get('name', 'N/A') for e in exercises]}")
            
            # Check errors
            errors = data.get('errors')
            if errors:
                print(f"⚠️  Errors reported: {errors}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - make sure services are running:")
            print("   - AgenticRAG: http://localhost:8000")
            print("   - DART: http://localhost:5001") 
            print("   - Orchestrator: http://localhost:8080")
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_service_health():
    """Check if all services are running."""
    print("\n🏥 Checking Service Health")
    print("=" * 30)
    
    services = {
        "AgenticRAG": "http://localhost:8000/health",
        "DART": "http://localhost:5001/health", 
        "Orchestrator": "http://localhost:8080/health"
    }
    
    all_healthy = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: OK")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                all_healthy = False
        except Exception as e:
            print(f"❌ {name}: {type(e).__name__}")
            all_healthy = False
    
    return all_healthy

if __name__ == "__main__":
    # First check service health
    if test_service_health():
        # Run integration tests
        test_visualize_motion_integration()
    else:
        print("\n❌ Some services are not running. Please start all services before testing.")
        print("\n📋 Startup order:")
        print("1. AgenticRAG: cd agenticRAG/agentic_rag_gemini && python api_server.py")
        print("2. DART: cd text-to-motion/DART && python api_server.py") 
        print("3. Orchestrator: cd agenticRAG/agentic_rag_gemini && python main_api.py")
