# Complete Python Programming Guide

## Table of Contents

1. [Python Fundamentals](#1-python-fundamentals)
2. [Data Structures](#2-data-structures)
3. [Control Flow & Functions](#3-control-flow--functions)
4. [Object-Oriented Programming](#4-object-oriented-programming)
5. [File Operations](#5-file-operations)
6. [Error Handling](#6-error-handling)
7. [Scientific Computing](#7-scientific-computing)
8. [Project Guidelines](#8-project-guidelines)
9. [Testing & Debugging](#9-testing--debugging)
10. [Best Practices](#10-best-practices)

## 1. Python Fundamentals

### Installation & Setup

```python
# Check Python version
python --version

# Python interpreter
>>> print("Hello World")
```

### Variables & Data Types

```python
# Numbers
integer = 42          # int
float_num = 3.14     # float
complex_num = 3 + 4j # complex

# Strings
single = 'Single quotes'
double = "Double quotes"
multiline = '''
Multiple
lines
'''

# String operations
name = "Python"
print(name.upper())      # PYTHON
print(name.lower())      # python
print(name[0:2])        # Py
print(len(name))        # 6
print(name + " Programming") # Python Programming

# Boolean
is_true = True
is_false = False

# None type
nothing = None

# Type conversion
str_num = "42"
int_num = int(str_num)    # String to integer
float_num = float(str_num) # String to float
str_back = str(int_num)   # Integer to string
```

## 2. Data Structures

### Lists

```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, [4, 5]]

# List operations
numbers.append(6)        # Add to end
numbers.insert(0, 0)    # Insert at index
numbers.remove(3)       # Remove value
popped = numbers.pop()  # Remove & return last item
numbers.extend([7, 8])  # Add multiple items
numbers.sort()          # Sort in place
numbers.reverse()       # Reverse in place

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

### Tuples

```python
# Creating tuples
coordinates = (3, 4)
single_item = (1,)      # Note the comma

# Tuple operations
x, y = coordinates      # Unpacking
nested = ((1, 2), (3, 4))
```

### Dictionaries

```python
# Creating dictionaries
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}

# Dictionary operations
person['email'] = 'john@example.com'  # Add/update
del person['age']                     # Delete
keys = person.keys()                  # Get keys
values = person.values()              # Get values
items = person.items()                # Get key-value pairs

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}
```

### Sets

```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
from_list = set([1, 2, 2, 3, 3, 4])  # Creates {1, 2, 3, 4}

# Set operations
unique_numbers.add(6)                 # Add item
unique_numbers.remove(1)              # Remove item
unique_numbers.discard(10)            # Remove if exists
union = set1 | set2                   # Union
intersection = set1 & set2            # Intersection
difference = set1 - set2              # Difference
```

## 3. Control Flow & Functions

### Conditional Statements

```python
# If statements
if condition:
    do_something()
elif other_condition:
    do_something_else()
else:
    do_default()

# Ternary operator
result = value_if_true if condition else value_if_false
```

### Loops

```python
# For loops
for i in range(5):
    print(i)

for item in collection:
    process(item)

# While loops
while condition:
    do_something()
    if exit_condition:
        break
    if skip_condition:
        continue

# Loop with else
for item in collection:
    if found_item:
        break
else:
    # Runs if loop completes without break
    not_found()
```

### Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Default parameters
def greet(name="World"):
    return f"Hello, {name}!"

# Multiple parameters
def power(base, exponent=2):
    return base ** exponent

# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

# Lambda functions
square = lambda x: x**2

# Decorators
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")
```

## 4. Object-Oriented Programming

### Classes and Objects

```python
class Person:
    # Class variable
    species = "Homo sapiens"

    # Constructor
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age

    # Instance method
    def greet(self):
        return f"Hello, I'm {self.name}"

    # Class method
    @classmethod
    def from_birth_year(cls, name, birth_year):
        return cls(name, 2024 - birth_year)

    # Static method
    @staticmethod
    def is_adult(age):
        return age >= 18

# Inheritance
class Employee(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id
```

### Special Methods

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __len__(self):
        return 2

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
```

## 5. File Operations

### File Handling

```python
# Reading files
with open('file.txt', 'r') as f:
    content = f.read()           # Read entire file
    lines = f.readlines()        # Read lines into list

# Writing files
with open('file.txt', 'w') as f:
    f.write('Hello\n')          # Write string
    f.writelines(['a\n', 'b\n']) # Write lines

# Appending to files
with open('file.txt', 'a') as f:
    f.write('New line\n')
```

### CSV Operations

```python
import csv

# Reading CSV
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        process(row)

# Writing CSV
with open('data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'age'])
    writer.writerows(data)
```

## 6. Error Handling

### Try-Except Blocks

```python
# Basic error handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Error: {e}")
else:
    print("No error occurred")
finally:
    print("This always runs")

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

## 7. Scientific Computing

### NumPy

```python
import numpy as np

# Arrays
arr = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])

# Array operations
arr + 1                  # Add to each element
arr * 2                  # Multiply each element
np.sqrt(arr)            # Square root
np.sum(arr)             # Sum of elements
np.mean(arr)            # Mean of elements

# Matrix operations
np.dot(matrix1, matrix2) # Matrix multiplication
np.transpose(matrix)     # Transpose
```

### Pandas

```python
import pandas as pd

# Creating DataFrames
df = pd.DataFrame({
    'name': ['John', 'Jane'],
    'age': [25, 30]
})

# Reading data
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Data operations
filtered = df[df['age'] > 25]
grouped = df.groupby('category').mean()
sorted_df = df.sort_values('age')
```

## 8. Project Guidelines

### Project Structure

```
project/
├── README.md
├── requirements.txt
├── setup.py
├── project_name/
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test_main.py
```

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## 9. Testing & Debugging

### Unit Testing

```python
import unittest

class TestMathFunctions(unittest.TestCase):
    def setUp(self):
        self.value = 10

    def test_addition(self):
        result = self.value + 5
        self.assertEqual(result, 15)

    def test_division(self):
        with self.assertRaises(ZeroDivisionError):
            self.value / 0

if __name__ == '__main__':
    unittest.main()
```

### Debugging

```python
# Print debugging
print(f"Variable value: {variable}")

# Using pdb
import pdb; pdb.set_trace()

# Using logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message")
logging.info("Info message")
logging.error("Error message")
```

## 10. Best Practices

### Code Style (PEP 8)

- Use 4 spaces for indentation
- Maximum line length of 79 characters
- Two blank lines before class definitions
- One blank line before function definitions
- Use snake_case for variables and functions
- Use CamelCase for class names

### Documentation

```python
def complex_function(param1, param2):
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this error occurs
    """
    pass
```

### Performance Tips

1. Use list comprehensions instead of loops when possible
2. Use generators for large sequences
3. Use built-in functions (map, filter, reduce)
4. Profile code using cProfile
5. Use appropriate data structures
6. Avoid global variables
7. Use 'join' for string concatenation

### Common Gotchas

1. Mutable default arguments
2. Late binding closures
3. Integer caching
4. Float precision issues
5. Iterator exhaustion

### Security Best Practices

1. Input validation
2. Use secrets module for cryptographic operations
3. Avoid eval() and exec()
4. Properly handle sensitive data
5. Use environment variables for secrets
