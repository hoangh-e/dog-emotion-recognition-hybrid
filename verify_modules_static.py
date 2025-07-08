#!/usr/bin/env python3
"""
Static Module Verification Script

This script verifies that all modules in dog_emotion_classification package have
the required functions by analyzing source code statically (without importing).

Author: Dog Emotion Recognition Team
Date: 2025
"""

import os
import re
from pathlib import Path


def analyze_python_file(file_path):
    """Analyze a Python file for function definitions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all function definitions
        function_pattern = r'^def\s+(\w+)\s*\('
        functions = re.findall(function_pattern, content, re.MULTILINE)
        
        # Find load functions
        load_functions = [f for f in functions if f.startswith('load_') and f.endswith('_model')]
        
        # Find predict functions
        predict_functions = [f for f in functions if f.startswith('predict_emotion_')]
        
        # Check emotion classes order
        emotion_classes_correct = False
        if "['angry', 'happy', 'relaxed', 'sad']" in content:
            emotion_classes_correct = True
        elif "'angry', 'happy', 'relaxed', 'sad'" in content:
            emotion_classes_correct = True
        
        return {
            'functions': functions,
            'load_functions': load_functions,
            'predict_functions': predict_functions,
            'emotion_classes_correct': emotion_classes_correct,
            'content_size': len(content)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'functions': [],
            'load_functions': [],
            'predict_functions': [],
            'emotion_classes_correct': False,
            'content_size': 0
        }


def get_function_signature_from_source(content, function_name):
    """Extract function signature from source code."""
    try:
        pattern = rf'^def\s+{re.escape(function_name)}\s*\([^)]*\):'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(0)
        return "Signature not found"
    except:
        return "Error extracting signature"


def verify_module_static(module_name):
    """Verify a module by static analysis."""
    print(f"\n{'='*60}")
    print(f"🔍 STATIC VERIFICATION: {module_name}")
    print(f"{'='*60}")
    
    # Find the module file
    module_file = Path(f"dog_emotion_classification/{module_name}.py")
    
    if not module_file.exists():
        print(f"❌ Module file not found: {module_file}")
        return {
            'module': module_name,
            'exists': False,
            'status': '❌ FILE NOT FOUND'
        }
    
    print(f"✅ Module file found: {module_file}")
    
    # Analyze the file
    analysis = analyze_python_file(module_file)
    
    if 'error' in analysis:
        print(f"❌ Error analyzing file: {analysis['error']}")
        return {
            'module': module_name,
            'exists': True,
            'error': analysis['error'],
            'status': '❌ ANALYSIS ERROR'
        }
    
    print(f"📊 File size: {analysis['content_size']} characters")
    print(f"📊 Total functions: {len(analysis['functions'])}")
    print(f"🔧 Load functions: {len(analysis['load_functions'])}")
    print(f"🎯 Predict functions: {len(analysis['predict_functions'])}")
    
    # Check for main functions
    expected_load_func = f"load_{module_name}_model"
    expected_predict_func = f"predict_emotion_{module_name}"
    
    has_main_load = expected_load_func in analysis['functions']
    has_main_predict = expected_predict_func in analysis['functions']
    
    if has_main_load:
        print(f"✅ Main load function found: {expected_load_func}")
    else:
        print(f"❌ Main load function missing: {expected_load_func}")
        # Check for alternatives
        alternatives = [f for f in analysis['load_functions'] if module_name in f]
        if alternatives:
            print(f"   Found alternatives: {alternatives}")
    
    if has_main_predict:
        print(f"✅ Main predict function found: {expected_predict_func}")
    else:
        print(f"❌ Main predict function missing: {expected_predict_func}")
        # Check for alternatives
        alternatives = [f for f in analysis['predict_functions'] if module_name in f]
        if alternatives:
            print(f"   Found alternatives: {alternatives}")
    
    # Check emotion classes
    if analysis['emotion_classes_correct']:
        print(f"✅ Correct emotion classes order found")
    else:
        print(f"⚠️  Emotion classes order may be incorrect or not found")
    
    # List all functions found
    if analysis['load_functions']:
        print(f"\n📋 All load functions:")
        for func in sorted(analysis['load_functions']):
            print(f"   • {func}")
    
    if analysis['predict_functions']:
        print(f"\n📋 All predict functions:")
        for func in sorted(analysis['predict_functions']):
            print(f"   • {func}")
    
    # Overall status
    if has_main_load and has_main_predict:
        status = "✅ COMPLETE"
    elif analysis['load_functions'] or analysis['predict_functions']:
        status = "⚠️  PARTIAL"
    else:
        status = "❌ INCOMPLETE"
    
    print(f"\n🎯 Module Status: {status}")
    
    return {
        'module': module_name,
        'exists': True,
        'has_main_load': has_main_load,
        'has_main_predict': has_main_predict,
        'load_functions': analysis['load_functions'],
        'predict_functions': analysis['predict_functions'],
        'emotion_classes_correct': analysis['emotion_classes_correct'],
        'total_functions': len(analysis['functions']),
        'status': status
    }


def main():
    """Main verification function."""
    print("🚀 DOG EMOTION CLASSIFICATION STATIC MODULE VERIFICATION")
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
        result = verify_module_static(module_name)
        results.append(result)
    
    # Summary report
    print(f"\n{'='*80}")
    print("📊 STATIC VERIFICATION SUMMARY REPORT")
    print(f"{'='*80}")
    
    complete_modules = [r for r in results if r.get('status') == '✅ COMPLETE']
    partial_modules = [r for r in results if r.get('status') == '⚠️  PARTIAL']
    incomplete_modules = [r for r in results if r.get('status') == '❌ INCOMPLETE']
    error_modules = [r for r in results if r.get('status') in ['❌ FILE NOT FOUND', '❌ ANALYSIS ERROR']]
    
    print(f"✅ Complete modules: {len(complete_modules)}/{len(modules)}")
    print(f"⚠️  Partial modules: {len(partial_modules)}")
    print(f"❌ Incomplete modules: {len(incomplete_modules)}")
    print(f"💥 Error modules: {len(error_modules)}")
    
    if complete_modules:
        print(f"\n✅ COMPLETE MODULES ({len(complete_modules)}):")
        for result in complete_modules:
            module = result['module']
            load_count = len(result.get('load_functions', []))
            predict_count = len(result.get('predict_functions', []))
            emotion_ok = "✅" if result.get('emotion_classes_correct') else "⚠️"
            print(f"   • {module}: {load_count} load, {predict_count} predict, emotions {emotion_ok}")
    
    if partial_modules:
        print(f"\n⚠️  PARTIAL MODULES ({len(partial_modules)}):")
        for result in partial_modules:
            module = result['module']
            issues = []
            if not result.get('has_main_load'):
                issues.append("missing main load")
            if not result.get('has_main_predict'):
                issues.append("missing main predict")
            print(f"   • {module}: {', '.join(issues)}")
    
    if incomplete_modules:
        print(f"\n❌ INCOMPLETE MODULES ({len(incomplete_modules)}):")
        for result in incomplete_modules:
            module = result['module']
            print(f"   • {module}: no load/predict functions found")
    
    if error_modules:
        print(f"\n💥 ERROR MODULES ({len(error_modules)}):")
        for result in error_modules:
            module = result['module']
            status = result.get('status', 'Unknown error')
            print(f"   • {module}: {status}")
    
    # Function statistics
    print(f"\n📊 FUNCTION STATISTICS:")
    total_load_functions = sum(len(r.get('load_functions', [])) for r in results if r.get('exists'))
    total_predict_functions = sum(len(r.get('predict_functions', [])) for r in results if r.get('exists'))
    total_functions = sum(r.get('total_functions', 0) for r in results if r.get('exists'))
    
    print(f"🔧 Total load functions across all modules: {total_load_functions}")
    print(f"🎯 Total predict functions across all modules: {total_predict_functions}")
    print(f"📊 Total functions across all modules: {total_functions}")
    
    # Emotion classes verification
    correct_emotion_modules = [r for r in results if r.get('emotion_classes_correct')]
    print(f"🏷️  Modules with correct emotion classes: {len(correct_emotion_modules)}/{len(results)}")
    
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
    results = main()
    
    # Save results to file
    output_file = "MODULE_VERIFICATION_SUMMARY.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Dog Emotion Classification Module Verification Summary\n\n")
        f.write("## Overview\n\n")
        
        complete_count = len([r for r in results if r.get('status') == '✅ COMPLETE'])
        partial_count = len([r for r in results if r.get('status') == '⚠️  PARTIAL'])
        total_count = len(results)
        completion_rate = complete_count / total_count * 100
        
        f.write(f"- **Total Modules**: {total_count}\n")
        f.write(f"- **Complete Modules**: {complete_count}\n")
        f.write(f"- **Partial Modules**: {partial_count}\n")
        f.write(f"- **Completion Rate**: {completion_rate:.1f}%\n\n")
        
        f.write("## Module Status\n\n")
        f.write("| Module | Status | Load Functions | Predict Functions | Emotion Classes |\n")
        f.write("|--------|--------|----------------|-------------------|----------------|\n")
        
        for result in sorted(results, key=lambda x: x['module']):
            module = result['module']
            status = result.get('status', 'Unknown')
            load_count = len(result.get('load_functions', []))
            predict_count = len(result.get('predict_functions', []))
            emotion_ok = "✅" if result.get('emotion_classes_correct') else "⚠️"
            f.write(f"| {module} | {status} | {load_count} | {predict_count} | {emotion_ok} |\n")
        
        f.write(f"\n## Verification Details\n\n")
        f.write("### Required Functions\n")
        f.write("Each module should have:\n")
        f.write("1. `load_{module}_model(model_path, num_classes=4, input_size, device='cuda')`\n")
        f.write("2. `predict_emotion_{module}(image_path, model, transform, head_bbox=None, device='cuda', emotion_classes=['angry', 'happy', 'relaxed', 'sad'])`\n\n")
        
        f.write("### Emotion Classes Order\n")
        f.write("All modules must use the correct emotion classes order:\n")
        f.write("`['angry', 'happy', 'relaxed', 'sad']`\n\n")
        
        f.write("### Verification Method\n")
        f.write("This verification was performed using static code analysis without importing dependencies.\n\n")
        
        f.write("### Verification Date\n")
        f.write("This verification was performed on: 2025\n")
    
    print(f"\n📄 Detailed results saved to: {output_file}") 