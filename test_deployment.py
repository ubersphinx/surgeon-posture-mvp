#!/usr/bin/env python3
"""
Test script to verify OpenCV headless installation works correctly
"""

def test_opencv_import():
    """Test if OpenCV can be imported successfully"""
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic functionality
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image.fill(128)
        
        # Test color conversion
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✅ Basic OpenCV operations work")
        
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_mediapipe_import():
    """Test if MediaPipe can be imported successfully"""
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"MediaPipe version: {mp.__version__}")
        
        # Test pose detection initialization
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe Pose initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_streamlit_import():
    """Test if Streamlit can be imported successfully"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing deployment dependencies...")
    print("=" * 50)
    
    opencv_ok = test_opencv_import()
    mediapipe_ok = test_mediapipe_import()
    streamlit_ok = test_streamlit_import()
    
    print("=" * 50)
    print("📊 Test Results:")
    print(f"OpenCV: {'✅ PASS' if opencv_ok else '❌ FAIL'}")
    print(f"MediaPipe: {'✅ PASS' if mediapipe_ok else '❌ FAIL'}")
    print(f"Streamlit: {'✅ PASS' if streamlit_ok else '❌ FAIL'}")
    
    if all([opencv_ok, mediapipe_ok, streamlit_ok]):
        print("\n🎉 All tests passed! Deployment should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the requirements.txt file.") 