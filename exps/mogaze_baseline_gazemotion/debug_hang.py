#!/usr/bin/env python3
"""
Enhanced debug script to investigate the model architecture and DCT operations
"""

if __name__ == '__main__':
    print("=== ENHANCED MODEL DEBUG SCRIPT ===")
    
    try:
        print("1. Importing and setting up...")
        import torch
        import numpy as np
        from mogaze_config import config
        from model import siMLPe as Model
        from exps.mogaze_baseline_3d.mogaze_dataset_3d import MoGazeDataset
        print("   ✓ All imports successful")
        
        print("2. Checking config values...")
        print(f"   h36m_input_length: {config.motion.h36m_input_length}")
        print(f"   h36m_input_length_dct: {config.motion.h36m_input_length_dct}")
        print(f"   h36m_target_length_train: {config.motion.h36m_target_length_train}")
        print(f"   motion.dim: {config.motion.dim}")
        print(f"   deriv_input: {config.deriv_input}")
        print(f"   deriv_output: {config.deriv_output}")
        
        print("3. Creating DCT matrices...")
        def get_dct_matrix(N):
            dct_m = np.eye(N)
            for k in np.arange(N):
                for i in np.arange(N):
                    w = np.sqrt(2 / N)
                    if k == 0:
                        w = np.sqrt(1 / N)
                    dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
            idct_m = np.linalg.inv(dct_m)
            return dct_m, idct_m
        
        dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
        dct_tensor = torch.tensor(dct_m).float().cuda().unsqueeze(0)
        idct_tensor = torch.tensor(idct_m).float().cuda().unsqueeze(0)
        print(f"   ✓ DCT matrix shape: {dct_tensor.shape}")
        print(f"   ✓ IDCT matrix shape: {idct_tensor.shape}")
        
        print("4. Creating model...")
        model = Model(config)
        model.cuda()
        print("   ✓ Model created and moved to GPU")
        
        print("5. Testing the full training pipeline step by step...")
        
        # Create test input matching training data
        batch_size = 2
        input_frames = 50
        target_frames = 10
        dims = 66
        
        test_input = torch.randn(batch_size, input_frames, dims).cuda()
        test_target = torch.randn(batch_size, target_frames, dims).cuda()
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Target shape: {test_target.shape}")
        
        print("6. Applying DCT (if deriv_input is True)...")
        if config.deriv_input:
            print("   Applying DCT transformation...")
            # This is exactly what train_step does
            b, n, c = test_input.shape
            test_input_dct = test_input.clone()
            test_input_dct = torch.matmul(dct_tensor[:, :, :config.motion.h36m_input_length], test_input_dct)
            print(f"   After DCT: {test_input_dct.shape}")
            model_input = test_input_dct
        else:
            model_input = test_input
            print("   No DCT applied")
        
        print("7. Forward pass through model...")
        with torch.no_grad():
            raw_output = model(model_input)
        print(f"   Raw model output shape: {raw_output.shape}")
        
        print("8. Applying IDCT (if deriv_input is True)...")
        if config.deriv_input:
            print("   Applying IDCT transformation...")
            # This is exactly what train_step does
            motion_pred = torch.matmul(idct_tensor[:, :config.motion.h36m_input_length, :], raw_output)
            print(f"   After IDCT: {motion_pred.shape}")
        else:
            motion_pred = raw_output
            print("   No IDCT applied")
        
        print("9. Applying residual offset (if deriv_output is True)...")
        if config.deriv_output:
            print("   Adding residual offset...")
            offset = test_input[:, -1:].cuda()
            print(f"   Offset shape: {offset.shape}")
            motion_pred = motion_pred[:, :config.motion.h36m_target_length_train] + offset
            print(f"   After residual + slice: {motion_pred.shape}")
        else:
            motion_pred = motion_pred[:, :config.motion.h36m_target_length_train]
            print(f"   After slice only: {motion_pred.shape}")
        
        print("10. Final output analysis...")
        print(f"    Final prediction shape: {motion_pred.shape}")
        print(f"    Target shape: {test_target.shape}")
        
        if motion_pred.shape[1] == test_target.shape[1]:
            print("    ✅ SUCCESS! Output matches target length")
            
            # Test loss calculation
            print("11. Testing loss calculation...")
            b, n, c = test_target.shape
            joints = 22
            motion_pred_reshaped = motion_pred.reshape(b, n, joints, 3).reshape(-1, 3)
            target_reshaped = test_target.reshape(b, n, joints, 3).reshape(-1, 3)
            loss = torch.mean(torch.norm(motion_pred_reshaped - target_reshaped, 2, 1))
            print(f"    ✅ Loss calculation successful: {loss.item():.6f}")
            
        else:
            print(f"    ❌ MISMATCH! Expected {test_target.shape[1]} frames, got {motion_pred.shape[1]}")
            
            print("\n=== DETAILED MODEL ARCHITECTURE ANALYSIS ===")
            print("Model structure:")
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # leaf modules
                    print(f"  {name}: {module}")
            
            print(f"\nModel config details:")
            print(f"  temporal_fc_in: {model.temporal_fc_in}")
            print(f"  temporal_fc_out: {model.temporal_fc_out}")
            
            # Check the specific linear layer dimensions
            print(f"\nLinear layer dimensions:")
            if hasattr(model, 'motion_fc_in'):
                print(f"  motion_fc_in: {model.motion_fc_in}")
            if hasattr(model, 'motion_fc_out'):
                print(f"  motion_fc_out: {model.motion_fc_out}")
        
        print("\n🎉 ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ ERROR FOUND!")
        print(f"Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== DEBUG COMPLETE ===")