# Python Scientific Computing Certification Study Notes

## 1. Python Basics

### Variables and Data Types

```python
# Numbers
integer_num = 42
float_num = 3.14

# Strings
text = "Hello World"
multiline = '''Multiple
line string'''

# Boolean
is_true = True
is_false = False

# Lists, Tuples, Sets, Dictionaries
my_list = [1, 2, 3]           # Mutable, ordered
my_tuple = (1, 2, 3)          # Immutable, ordered
my_set = {1, 2, 3}            # Mutable, unordered, unique
my_dict = {'a': 1, 'b': 2}    # Key-value pairs
```

### Control Flow

```python
# If statements
if condition:
    do_something()
elif other_condition:
    do_something_else()
else:
    do_default()

# Loops
for item in sequence:
    process(item)

while condition:
    do_something()
```

## 2. Functions and Modules

### Function Definition

```python
def function_name(parameter1, parameter2=default_value):
    """Docstring explaining function purpose"""
    # Function body
    return result

# Lambda functions
square = lambda x: x**2
```

### Modules and Imports

```python
import module_name
from module_name import function_name
from module_name import *  # Not recommended
```

## 3. Object-Oriented Programming

```python
class ClassName:
    def __init__(self, parameter):
        self.attribute = parameter

    def method(self):
        return self.attribute

# Inheritance
class Child(Parent):
    def __init__(self):
        super().__init__()
```

## 4. Scientific Computing Concepts

### NumPy Basics

```python
import numpy as np

# Arrays
array = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])

# Operations
sum = np.sum(array)
mean = np.mean(array)
```

### Working with Data

```python
import pandas as pd

# Reading data
df = pd.read_csv('file.csv')

# Basic operations
mean = df['column'].mean()
filtered = df[df['column'] > value]
```

## 5. Project Requirements

### Arithmetic Formatter

- Function to format arithmetic problems
- Handle addition and subtraction
- Proper spacing and alignment

### Time Calculator

- Add duration to start time
- Handle day changes
- Format output correctly

### Budget App

- Create budget categories
- Track deposits/withdrawals
- Generate spending chart

### Polygon Area Calculator

- Rectangle and Square classes
- Calculate area, perimeter
- String representation

### Probability Calculator

- Hat class for balls
- Calculate experiment probability
- Handle random draws

## Common Debugging Tips

```python
# Print debugging
print(f"Variable value: {variable}")

# Using debugger
import pdb; pdb.set_trace()

# Exception handling
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")
```

## Testing

```python
import unittest

class TestClass(unittest.TestCase):
    def test_function(self):
        self.assertEqual(actual, expected)
        self.assertTrue(condition)
```

## Best Practices

1. Follow PEP 8 style guide
2. Write descriptive variable names
3. Comment complex logic
4. Use docstrings
5. Test your code
6. Handle errors gracefully
7. Keep functions small and focused

## Common Libraries

1. NumPy - Numerical computing
2. Pandas - Data manipulation
3. Matplotlib - Plotting
4. SciPy - Scientific computing
5. unittest - Testing
