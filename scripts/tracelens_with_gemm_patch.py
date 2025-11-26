#!/usr/bin/env python3
# TraceLens GEMM patch - adds GEMM sheet for compiled PyTorch traces
import re
import sys
import pandas as pd


def apply_gemm_patches():
    print("Applying TraceLens GEMM patches...")
    
    # Patch the main generate_perf_report function
    try:
        from TraceLens.Reporting.generate_perf_report_pytorch import main as original_main
        
        def patched_main():
            # Import here to avoid circular imports
            import pandas as pd
            from TraceLens.Reporting import generate_perf_report_pytorch
            
            # Save original write function
            original_write = pd.ExcelWriter.__enter__
            
            def patched_excel_enter(self):
                result = original_write(self)
                
                # Hook into DataFrame writing
                original_to_excel = pd.DataFrame.to_excel
                
                def patched_to_excel(df, excel_writer, sheet_name=None, **kwargs):
                    # Write original sheet
                    original_to_excel(df, excel_writer, sheet_name, **kwargs)
                    
                    # If this is ops_summary, also create GEMM sheet
                    if sheet_name == 'ops_summary' and 'name' in df.columns:
                        # Find ops that use GEMM kernels
                        gemm_ops = []
                        
                        # Check for CompiledFunction (torch.compile uses GEMM)
                        if 'CompiledFunction' in df['name'].values:
                            gemm_ops.append(df[df['name'] == 'CompiledFunction'])
                        if 'CompiledFunctionBackward' in df['name'].values:
                            gemm_ops.append(df[df['name'] == 'CompiledFunctionBackward'])
                        
                        # Check for explicit matmul ops
                        matmul_mask = df['name'].str.contains('mm|matmul|bmm|addmm|linear', case=False, na=False)
                        if matmul_mask.any():
                            gemm_ops.append(df[matmul_mask])
                        
                        if gemm_ops:
                            gemm_df = pd.concat(gemm_ops, ignore_index=True)
                            # Add GEMM category column
                            gemm_df['gemm_category'] = 'GEMM'
                            # Write GEMM sheet
                            print(f"  Creating GEMM sheet with {len(gemm_df)} ops")
                            original_to_excel(gemm_df, excel_writer, 'GEMM', **kwargs)
                
                pd.DataFrame.to_excel = patched_to_excel
                return result
            
            pd.ExcelWriter.__enter__ = patched_excel_enter
            
            # Run original main
            return original_main()
        
        # Replace main in the module
        sys.modules['TraceLens.Reporting.generate_perf_report_pytorch'].main = patched_main
        
        print("  [OK] Patched report generation")
    except Exception as e:
        print(f"  [ERROR] Could not patch: {e}")
        import traceback
        traceback.print_exc()
    
    print("[OK] GEMM patches applied!\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: tracelens_with_gemm_patch.py <command> [args...]")
        sys.exit(1)
    
    # Apply patches first
    apply_gemm_patches()
    
    # Import after patching
    from TraceLens.Reporting.generate_perf_report_pytorch import main as generate_perf_report_main
    from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import main as generate_multi_rank_collective_report_main
    from TraceLens.Reporting.compare_perf_reports_pytorch import main as compare_perf_reports_main
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "generate_perf_report":
        generate_perf_report_main()
    elif command == "generate_multi_rank_collective":
        generate_multi_rank_collective_report_main()
    elif command == "compare_perf_reports":
        compare_perf_reports_main()
    else:
        print(f"Error: Unknown command '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()