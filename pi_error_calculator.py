"""
PI Error Calculator

Utility functions to calculate error metrics between a given value and the mathematical constant PI.
"""

import numpy as np


def calculate_pi_error(current_value):
    """
    Calculate various error metrics between a current value and the mathematical constant PI.
    
    Parameters:
    -----------
    current_value : float
        The value to compare against PI
    
    Returns:
    --------
    dict
        Dictionary containing multiple error metrics:
        - pi_value: The actual value of PI (np.pi)
        - current_value: The input value for reference
        - absolute_error: |current_value - π|
        - relative_error: (|current_value - π| / π) * 100 (in percentage)
        - squared_error: (current_value - π)²
        - percent_accuracy: 100 - relative_error (how close to 100% accurate)
    
    Examples:
    ---------
    >>> result = calculate_pi_error(3.14)
    >>> print(f"Absolute error: {result['absolute_error']:.6f}")
    Absolute error: 0.001593
    
    >>> result = calculate_pi_error(3.14159265)
    >>> print(f"Relative error: {result['relative_error']:.8f}%")
    Relative error: 0.00000003%
    """
    pi_value = np.pi
    
    # Calculate different error metrics
    absolute_error = abs(current_value - pi_value)
    relative_error = (absolute_error / pi_value) * 100
    squared_error = (current_value - pi_value) ** 2
    percent_accuracy = 100 - relative_error
    
    return {
        'pi_value': pi_value,
        'current_value': current_value,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'squared_error': squared_error,
        'percent_accuracy': percent_accuracy
    }


def print_pi_error_analysis(current_value):
    """
    Print a detailed analysis of the error between current_value and PI.
    
    Parameters:
    -----------
    current_value : float
        The value to compare against PI
    
    Returns:
    --------
    dict
        The error metrics dictionary from calculate_pi_error()
    
    Examples:
    ---------
    >>> print_pi_error_analysis(3.14159)
    ================================================
    📊 PI ERROR ANALYSIS
    ================================================
    π (PI) value: 3.1415926536
    Current value: 3.1415900000
    
    📈 Error Metrics:
    Absolute error: 0.0000026536
    Relative error: 0.000084%
    Squared error: 0.0000000070
    Accuracy: 99.999916%
    
    ✅ Excellent precision! (error < 0.0001)
    ================================================
    """
    result = calculate_pi_error(current_value)
    
    print(f"{'='*50}")
    print(f"📊 PI ERROR ANALYSIS")
    print(f"{'='*50}")
    print(f"π (PI) value: {result['pi_value']:.10f}")
    print(f"Current value: {result['current_value']:.10f}")
    print(f"\n📈 Error Metrics:")
    print(f"Absolute error: {result['absolute_error']:.10f}")
    print(f"Relative error: {result['relative_error']:.6f}%")
    print(f"Squared error: {result['squared_error']:.10f}")
    print(f"Accuracy: {result['percent_accuracy']:.6f}%")
    
    # Error classification
    if result['absolute_error'] < 0.0001:
        print(f"\n✅ Excellent precision! (error < 0.0001)")
    elif result['absolute_error'] < 0.001:
        print(f"\n✓ Very good precision (error < 0.001)")
    elif result['absolute_error'] < 0.01:
        print(f"\n⚠️ Good precision (error < 0.01)")
    elif result['absolute_error'] < 0.1:
        print(f"\n⚠️ Acceptable precision (error < 0.1)")
    else:
        print(f"\n❌ Poor precision (error ≥ 0.1)")
    
    print(f"{'='*50}\n")
    
    return result


def compare_multiple_values_to_pi(values_list, print_results=True):
    """
    Compare multiple values to PI and return their error metrics.
    
    Parameters:
    -----------
    values_list : list of float
        List of values to compare against PI
    print_results : bool, optional
        Whether to print a summary table (default: True)
    
    Returns:
    --------
    list of dict
        List of error metric dictionaries for each value
    
    Examples:
    ---------
    >>> approximations = [3.0, 3.14, 3.14159, 3.141592653589793]
    >>> results = compare_multiple_values_to_pi(approximations)
    """
    results = []
    
    for value in values_list:
        result = calculate_pi_error(value)
        results.append(result)
    
    if print_results:
        print(f"{'='*80}")
        print(f"📊 COMPARISON OF MULTIPLE PI APPROXIMATIONS")
        print(f"{'='*80}")
        print(f"{'Value':<20} {'Absolute Error':<20} {'Relative Error (%)':<20} {'Accuracy (%)':<20}")
        print(f"{'-'*80}")
        
        for result in results:
            print(f"{result['current_value']:<20.10f} "
                  f"{result['absolute_error']:<20.10f} "
                  f"{result['relative_error']:<20.8f} "
                  f"{result['percent_accuracy']:<20.8f}")
        
        print(f"{'='*80}\n")
        
        # Find best approximation
        best_idx = min(range(len(results)), key=lambda i: results[i]['absolute_error'])
        print(f"🏆 Best approximation: {results[best_idx]['current_value']:.10f}")
        print(f"   with error: {results[best_idx]['absolute_error']:.10f}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("✓ PI Error Calculator Module\n")
    
    # Test with common PI approximations
    print("Example 1: Testing common approximations")
    approximations = [
        3.0,           # Ancient approximation
        3.14,          # Two decimal places
        3.14159,       # Five decimal places
        22/7,          # Archimedes' approximation
        355/113,       # Zu Chongzhi's approximation
        np.pi          # NumPy's PI (reference)
    ]
    
    compare_multiple_values_to_pi(approximations)
    
    # Test individual value
    print("\nExample 2: Detailed analysis of single value")
    print_pi_error_analysis(3.14159265)