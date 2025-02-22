import unittest
from dl_cm.scripts.common_scripts import chain_decorators

class TestCommonScripts(unittest.TestCase):
    def test_chain_decorators(self):
        # Define two simple decorators
        def double_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result *   2
            return wrapper

        def increment_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result +   1
            return wrapper

        # Create a function to apply the decorators to
        def multiply_by_three(number):
            return number *   3

        # Apply the decorators using chain_decorators
        decorated_func = chain_decorators(double_decorator, increment_decorator)(multiply_by_three)

        # Call the decorated function with an argument
        result = decorated_func(2)

        # Verify that the decorators were applied in the correct order
        # and that the final result is as expected
        expected_result = ((2 *  3) + 1) * 2  # Expected result after applying both decorators
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()