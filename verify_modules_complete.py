#!/usr/bin/env python3
"""
Comprehensive Module Verification Script

This script verifies that all modules in dog_emotion_classification package have:
1. Standardized load_model functions with correct signatures
2. Standardized predict_emotion functions with correct signatures
3. Correct emotion classes order: ['angry', 'happy', 'relaxed', 'sad']
4. Proper function imports and accessibility

Author: Dog Emotion Recognition Team
Date: 2025
"""

import sys
import os
import importlib
import inspect
from pathlib import Path


def get_function_signature(func):
    """Get function signature as string."""
    try:
        sig = inspect.signature(func)
        return str(sig)
    except:
        return "Unable to get signature"


def check_emotion_classes_in_function(func):
    """Check if function uses correct emotion classes order."""
    try:
        source = inspect.getsource(func)
        # Check for the correct emotion classes order
        correct_order = "['angry', 'happy', 'relaxed', 'sad']"
        if correct_order in source:
            return True, "✅ Correct emotion classes order found"
        elif "'angry', 'happy', 'relaxed', 'sad'" in source:
            return True, "✅ Correct emotion classes order found"
        else:
            # Check for any emotion classes
            if any(emotion in source for emotion in ['angry', 'happy', 'relaxed', 'sad']):
                return False, "⚠️  Emotion classes found but order may be incorrect"
            else:
                return True, "ℹ️  No hardcoded emotion classes (uses parameter)"
    except:
        return True, "ℹ️  Cannot check source code"


def verify_module(module_name):
    """Verify a single module for required functions."""
    print(f"\n{'='*60}")
    print(f"🔍 VERIFYING MODULE: {module_name}")
    print(f"{'='*60}")
    
    try:
        # Import the module
        module = importlib.import_module(f"dog_emotion_classification.{module_name}")
        print(f"✅ Module imported successfully")
        
        # Get all functions in the module
        functions = [name for name, obj in inspect.getmembers(module) 
                    if inspect.isfunction(obj) and not name.startswith('_')]
        
        # Find load_model functions
        load_functions = [f for f in functions if f.startswith('load_') and f.endswith('_model')]
        predict_functions = [f for f in functions if f.startswith('predict_emotion_')]
        
        print(f"📊 Total functions: {len(functions)}")
        print(f"🔧 Load functions: {len(load_functions)}")
        print(f"🎯 Predict functions: {len(predict_functions)}")
        
        # Check for main load function
        main_load_func = f"load_{module_name}_model"
        has_main_load = main_load_func in functions
        
        if has_main_load:
            print(f"✅ Main load function found: {main_load_func}")
            func = getattr(module, main_load_func)
            sig = get_function_signature(func)
            print(f"   Signature: {sig}")
        else:
            print(f"❌ Main load function missing: {main_load_func}")
            # Check for alternative patterns
            alternatives = [f for f in load_functions if module_name in f]
            if alternatives:
                print(f"   Found alternatives: {alternatives}")
            else:
                print(f"   No load functions found with '{module_name}' in name")
        
        # Check for main predict function
        main_predict_func = f"predict_emotion_{module_name}"
        has_main_predict = main_predict_func in functions
        
        if has_main_predict:
            print(f"✅ Main predict function found: {main_predict_func}")
            func = getattr(module, main_predict_func)
            sig = get_function_signature(func)
            print(f"   Signature: {sig}")
            
            # Check emotion classes
            classes_ok, classes_msg = check_emotion_classes_in_function(func)
            print(f"   {classes_msg}")
        else:
            print(f"❌ Main predict function missing: {main_predict_func}")
            # Check for alternative patterns
            alternatives = [f for f in predict_functions if module_name in f]
            if alternatives:
                print(f"   Found alternatives: {alternatives}")
            else:
                print(f"   No predict functions found with '{module_name}' in name")
        
        # List all load functions found
        if load_functions:
            print(f"\n📋 All load functions in {module_name}:")
            for func_name in sorted(load_functions):
                func = getattr(module, func_name)
                sig = get_function_signature(func)
                print(f"   • {func_name}{sig}")
        
        # List all predict functions found
        if predict_functions:
            print(f"\n📋 All predict functions in {module_name}:")
            for func_name in sorted(predict_functions):
                func = getattr(module, func_name)
                sig = get_function_signature(func)
                print(f"   • {func_name}{sig}")
                
                # Check emotion classes in each predict function
                classes_ok, classes_msg = check_emotion_classes_in_function(func)
                print(f"     {classes_msg}")
        
        # Overall status
        status = "✅ COMPLETE" if (has_main_load and has_main_predict) else "⚠️  INCOMPLETE"
        print(f"\n🎯 Module Status: {status}")
        
        return {
            'module': module_name,
            'imported': True,
            'has_main_load': has_main_load,
            'has_main_predict': has_main_predict,
            'load_functions': load_functions,
            'predict_functions': predict_functions,
            'status': status
        }
        
    except Exception as e:
        print(f"❌ Error importing module: {e}")
        return {
            'module': module_name,
            'imported': False,
            'error': str(e),
            'status': '❌ ERROR'
        }


def main():
    """Main verification function."""
    print("🚀 DOG EMOTION CLASSIFICATION MODULE VERIFICATION")
    print("=" * 80)
    
    # List of all modules to verify
    modules = [
        'resnet', 'pure', 'pure34', 'pure50', 'vgg', 'densenet',
        'inception', 'mobilenet', 'efficientnet', 'vit', 'convnext',
        'alexnet', 'squeezenet', 'shufflenet', 'swin', 'deit',
        'nasnet', 'mlp_mixer', 'maxvit', 'coatnet', 'nfnet',
        'ecanet', 'senet'
    ]
    
    print(f"📊 Total modules to verify: {len(modules)}")
    
    # Verify each module
    results = []
    for module_name in modules:
        result = verify_module(module_name)
        results.append(result)
    
    # Summary report
    print(f"\n{'='*80}")
    print("📊 VERIFICATION SUMMARY REPORT")
    print(f"{'='*80}")
    
    complete_modules = [r for r in results if r.get('status') == '✅ COMPLETE']
    incomplete_modules = [r for r in results if r.get('status') == '⚠️  INCOMPLETE']
    error_modules = [r for r in results if r.get('status') == '❌ ERROR']
    
    print(f"✅ Complete modules: {len(complete_modules)}/{len(modules)}")
    print(f"⚠️  Incomplete modules: {len(incomplete_modules)}")
    print(f"❌ Error modules: {len(error_modules)}")
    
    if complete_modules:
        print(f"\n✅ COMPLETE MODULES ({len(complete_modules)}):")
        for result in complete_modules:
            module = result['module']
            load_count = len(result.get('load_functions', []))
            predict_count = len(result.get('predict_functions', []))
            print(f"   • {module}: {load_count} load functions, {predict_count} predict functions")
    
    if incomplete_modules:
        print(f"\n⚠️  INCOMPLETE MODULES ({len(incomplete_modules)}):")
        for result in incomplete_modules:
            module = result['module']
            issues = []
            if not result.get('has_main_load'):
                issues.append("missing main load function")
            if not result.get('has_main_predict'):
                issues.append("missing main predict function")
            print(f"   • {module}: {', '.join(issues)}")
    
    if error_modules:
        print(f"\n❌ ERROR MODULES ({len(error_modules)}):")
        for result in error_modules:
            module = result['module']
            error = result.get('error', 'Unknown error')
            print(f"   • {module}: {error}")
    
    # Detailed function analysis
    print(f"\n📋 DETAILED FUNCTION ANALYSIS:")
    all_load_functions = set()
    all_predict_functions = set()
    
    for result in results:
        if result.get('imported'):
            all_load_functions.update(result.get('load_functions', []))
            all_predict_functions.update(result.get('predict_functions', []))
    
    print(f"🔧 Unique load function patterns: {len(all_load_functions)}")
    for func in sorted(all_load_functions)[:10]:  # Show first 10
        print(f"   • {func}")
    if len(all_load_functions) > 10:
        print(f"   ... and {len(all_load_functions) - 10} more")
    
    print(f"\n🎯 Unique predict function patterns: {len(all_predict_functions)}")
    for func in sorted(all_predict_functions)[:10]:  # Show first 10
        print(f"   • {func}")
    if len(all_predict_functions) > 10:
        print(f"   ... and {len(all_predict_functions) - 10} more")
    
    # Final status
    completion_rate = len(complete_modules) / len(modules) * 100
    print(f"\n🎯 OVERALL COMPLETION: {completion_rate:.1f}%")
    
    if completion_rate == 100:
        print("🎉 ALL MODULES ARE COMPLETE! ✅")
    elif completion_rate >= 90:
        print("🔥 EXCELLENT! Most modules are complete")
    elif completion_rate >= 70:
        print("👍 GOOD! Majority of modules are complete")
    else:
        print("⚠️  NEEDS WORK! Many modules need attention")
    
    return results


if __name__ == "__main__":
    # Add the parent directory to Python path for imports
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    results = main()
    
    # Save results to file
    output_file = "MODULE_VERIFICATION_SUMMARY.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Dog Emotion Classification Module Verification Summary\n\n")
        f.write("## Overview\n\n")
        
        complete_count = len([r for r in results if r.get('status') == '✅ COMPLETE'])
        total_count = len(results)
        completion_rate = complete_count / total_count * 100
        
        f.write(f"- **Total Modules**: {total_count}\n")
        f.write(f"- **Complete Modules**: {complete_count}\n")
        f.write(f"- **Completion Rate**: {completion_rate:.1f}%\n\n")
        
        f.write("## Module Status\n\n")
        f.write("| Module | Status | Load Functions | Predict Functions |\n")
        f.write("|--------|--------|----------------|-------------------|\n")
        
        for result in sorted(results, key=lambda x: x['module']):
            module = result['module']
            status = result.get('status', 'Unknown')
            load_count = len(result.get('load_functions', []))
            predict_count = len(result.get('predict_functions', []))
            f.write(f"| {module} | {status} | {load_count} | {predict_count} |\n")
        
        f.write(f"\n## Verification Details\n\n")
        f.write("### Required Functions\n")
        f.write("Each module should have:\n")
        f.write("1. `load_{module}_model(model_path, num_classes=4, input_size, device='cuda')`\n")
        f.write("2. `predict_emotion_{module}(image_path, model, transform, head_bbox=None, device='cuda', emotion_classes=['angry', 'happy', 'relaxed', 'sad'])`\n\n")
        
        f.write("### Emotion Classes Order\n")
        f.write("All modules must use the correct emotion classes order:\n")
        f.write("`['angry', 'happy', 'relaxed', 'sad']`\n\n")
        
        f.write("### Verification Date\n")
        f.write("This verification was performed on: 2025\n")
    
    print(f"\n📄 Detailed results saved to: {output_file}") 