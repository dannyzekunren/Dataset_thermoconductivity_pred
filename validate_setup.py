#!/usr/bin/env python3
"""
Validation script for Apple Silicon GPU-Accelerated Attention Hybrid BNN
Tests device setup, model loading, and basic functionality
"""

import torch
import sys
import json
import os
from pathlib import Path

def validate_environment():
    """Validate the environment setup"""
    print("🔍 Validating Environment Setup...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check device availability
    devices = []
    if torch.cuda.is_available():
        devices.append("CUDA GPU")
    if torch.backends.mps.is_available():
        devices.append("Apple Silicon GPU (MPS)")
    devices.append("CPU")
    
    print(f"Available devices: {', '.join(devices)}")
    
    # Select optimal device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Selected device: {device}")
    return device

def validate_files():
    """Check if all required files are present"""
    print("\n📁 Validating File Structure...")
    
    required_files = [
        "attention_hybrid_bnn_simple.py",
        "attention_hybrid_bnn_optimized.py", 
        "final_direct_optimization.py",
        "config/FINAL_best_attention_config.json",
        "results/FINAL_OPTIMIZED_attention_hybrid_results.csv",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("\n✅ All required files present!")
    return True

def validate_config():
    """Validate the configuration file"""
    print("\n⚙️ Validating Configuration...")
    
    try:
        with open("config/FINAL_best_attention_config.json", 'r') as f:
            config = json.load(f)
        
        # Check if we have best_hyperparameters section
        if "best_hyperparameters" in config:
            hyperparams = config["best_hyperparameters"]
        else:
            hyperparams = config
        
        required_keys = [
            "hidden_dim", "embedding_dim", "num_heads", 
            "dropout", "learning_rate", "weight_decay", "batch_size"
        ]
        
        for key in required_keys:
            if key not in hyperparams:
                print(f"❌ Missing config key: {key}")
                return False
            print(f"✅ {key}: {hyperparams[key]}")
        
        # Show performance if available
        if "performance" in config:
            print(f"✅ Performance data available")
            perf = config["performance"]
            if "random_split" in perf:
                print(f"   Random Split R²: {perf['random_split']['r2']}")
        
        print("\n✅ Configuration valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_model_import():
    """Test if the model can be imported"""
    print("\n🧠 Testing Model Import...")
    
    try:
        sys.path.append('.')
        import attention_hybrid_bnn_simple
        print("✅ Simple model import successful!")
        
        import attention_hybrid_bnn_optimized  
        print("✅ Optimized model import successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_device_performance():
    """Test basic device performance"""
    print("\n🚀 Testing Device Performance...")
    
    device = validate_environment()
    
    # Test basic tensor operations
    try:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        import time
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        print(f"✅ Matrix multiplication test passed!")
        print(f"   Time: {end_time - start_time:.4f} seconds")
        print(f"   Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Device performance test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🍎 Apple Silicon GPU-Accelerated Attention Hybrid BNN Validation")
    print("=" * 70)
    
    tests = [
        ("Environment", validate_environment),
        ("File Structure", validate_files),
        ("Configuration", validate_config),
        ("Model Import", test_model_import),
        ("Device Performance", test_device_performance)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Validation Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Model is ready to use!")
        print("\n🚀 Quick Start:")
        print("   python attention_hybrid_bnn_simple.py")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
