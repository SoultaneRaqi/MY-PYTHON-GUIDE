# COMPLETE PYTHON REFERENCE GUIDE

#THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)


# 1. BASIC DATA TYPES AND VARIABLES
# Numbers
integer_num = 42          # Integer (whole number)
float_num = 3.14         # Float (decimal number)
complex_num = 1 + 2j     # Complex number
hex_num = 0xFF          # Hexadecimal number
oct_num = 0o777         # Octal number
bin_num = 0b1010        # Binary number

# Strings
single_quotes = 'Hello'  # String with single quotes
double_quotes = "World"  # String with double quotes
multi_line = """This is
a multi-line
string"""              # Multi-line string
raw_string = r"C:\new" # Raw string (backslashes not escaped)

# String Methods
text = "hello, world"
print(text.capitalize())  # Capitalize first letter
print(text.upper())      # Convert to uppercase
print(text.lower())      # Convert to lowercase
print(text.strip())      # Remove whitespace
print(text.split(','))   # Split string into list
print(text.replace('o', '0'))  # Replace characters
print(text.find('world'))      # Find substring
print(text.count('l'))         # Count occurrences

# Boolean
is_true = True
is_false = False
none_value = None

# 2. ADVANCED DATA STRUCTURES
# Lists (Mutable sequences)
my_list = [1, 2, 3, 4, 5]
my_list.append(6)        # Add item to end
my_list.extend([7, 8])   # Add multiple items
my_list.insert(0, 0)     # Insert at position
my_list.remove(3)        # Remove first occurrence
my_list.pop()            # Remove and return last item
my_list.sort()           # Sort in place
my_list.reverse()        # Reverse in place
sliced = my_list[1:4]    # Slice list
stepped = my_list[::2]   # Step through list

# Tuples (Immutable sequences)
my_tuple = (1, 2, 3)
single_tuple = (1,)      # Single item tuple needs comma
nested_tuple = (1, (2, 3), 4)
x, y, z = my_tuple       # Tuple unpacking

# Dictionaries (Key-value mappings)
my_dict = {
    'name': 'John',
    'age': 30,
    'skills': ['Python', 'JavaScript']
}
my_dict.update({'email': 'john@example.com'})  # Add/update multiple
my_dict.get('name', 'Unknown')     # Get with default
my_dict.setdefault('city', 'NY')   # Set if key doesn't exist
dict_keys = my_dict.keys()         # Get keys view
dict_values = my_dict.values()     # Get values view
dict_items = my_dict.items()       # Get items view

# Sets (Unordered unique elements)
my_set = {1, 2, 3, 4, 5}
my_set.add(6)            # Add single item
my_set.update({7, 8, 9}) # Add multiple items
my_set.remove(5)         # Remove item (raises error if missing)
my_set.discard(5)        # Remove item (no error if missing)
set2 = {4, 5, 6}
union = my_set | set2    # Union of sets
intersect = my_set & set2  # Intersection
diff = my_set - set2     # Difference
sym_diff = my_set ^ set2 # Symmetric difference

# 3. CONTROL FLOW AND LOOPS
# If statements with complex conditions
age = 25
has_license = True
if age >= 18 and has_license:
    print("Can drive")
elif age >= 16:
    print("Can get license")
else:
    print("Too young")

# Match statement (Python 3.10+)
status = 'ready'
match status:
    case 'ready':
        print("System is ready")
    case 'busy':
        print("System is busy")
    case _:
        print("Unknown status")

# For loops with enumerate and zip
colors = ['red', 'green', 'blue']
numbers = [1, 2, 3]
for i, color in enumerate(colors):
    print(f"Color {i}: {color}")

for color, number in zip(colors, numbers):
    print(f"{color}: {number}")

# While loops with else
count = 0
while count < 5:
    print(count)
    count += 1
else:
    print("Loop completed normally")

# 4. FUNCTIONS AND LAMBDA EXPRESSIONS
# Function with type hints (Python 3.5+)
def calculate_total(
    prices: list[float],
    tax_rate: float = 0.1,
    *args: tuple,
    **kwargs: dict
) -> float:
    """
    Calculate total price including tax.
    
    Args:
        prices: List of prices
        tax_rate: Tax rate as decimal
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments
    
    Returns:
        Total price including tax
    """
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    return subtotal + tax

# Lambda functions with map, filter, reduce
from functools import reduce
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
product = reduce(lambda x, y: x * y, numbers)

# Decorators
def timing_decorator(func):
    from time import time
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Function took {end-start} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    import time
    time.sleep(1)

# 5. CLASSES AND OBJECT-ORIENTED PROGRAMMING
class Animal:
    # Class variable
    species_count = 0
    
    def __init__(self, name: str):
        # Instance variable
        self.name = name
        Animal.species_count += 1
    
    # Instance method
    def speak(self) -> str:
        raise NotImplementedError
    
    # Class method
    @classmethod
    def get_species_count(cls) -> int:
        return cls.species_count
    
    # Static method
    @staticmethod
    def is_mammal(species: str) -> bool:
        return species in ["dog", "cat", "elephant"]
    
    # Property decorator
    @property
    def name_upper(self) -> str:
        return self.name.upper()

# Inheritance
class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name)
        self.breed = breed
    
    def speak(self) -> str:
        return f"{self.name} says Woof!"

# 6. EXCEPTION HANDLING AND CONTEXT MANAGERS
# Custom exception
class CustomError(Exception):
    pass

# Try-except with multiple exceptions
def divide_numbers(a: float, b: float) -> float:
    try:
        result = a / b
    except ZeroDivisionError:
        raise CustomError("Cannot divide by zero!")
    except TypeError:
        print("Invalid types")
        raise
    else:
        print("Division successful")
        return result
    finally:
        print("Cleanup code")

# Context manager
class FileManager:
    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# 7. MODULES AND PACKAGES
# Example module structure
"""
my_package/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        module3.py
"""

# Import variations
import math
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np  # Common alias
from .module1 import function1  # Relative import

# 8. FILE AND DATA HANDLING
# File operations
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = f.readlines()
    f.seek(0)  # Return to start of file

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('Hello\n')
    f.writelines(['Line 1\n', 'Line 2\n'])

# JSON handling
import json
data = {'name': 'John', 'age': 30}
json_str = json.dumps(data, indent=2)
parsed_data = json.loads(json_str)

# CSV handling
import csv
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age'])
    writer.writerows([['John', 30], ['Jane', 25]])

# 9. GENERATORS AND ITERATORS
# Generator function
def fibonacci(n: int):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Generator expression
squares_gen = (x**2 for x in range(10))

# Custom iterator
class CountUp:
    def __init__(self, start: int, end: int):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        current = self.current
        self.current += 1
        return current

# 10. ASYNC PROGRAMMING
import asyncio

async def fetch_data(url: str) -> str:
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network delay
    return f"Data from {url}"

async def main():
    urls = [
        'http://example.com/1',
        'http://example.com/2'
    ]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# 11. TYPE HINTS AND ANNOTATIONS
from typing import TypeVar, Generic, Union, Any

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> Optional[T]:
        return self.items.pop() if self.items else None

# 12. DEBUGGING AND TESTING
import unittest
import logging
import pytest

# Unit testing
class TestMathFunctions(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)
    
    def test_division(self):
        with self.assertRaises(ZeroDivisionError):
            1 / 0

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Application started")
logger.error("An error occurred")

# 13. STANDARD LIBRARY HIGHLIGHTS
from collections import defaultdict, Counter, namedtuple
from datetime import datetime, timedelta
from itertools import chain, combinations
from functools import partial, lru_cache
from pathlib import Path
import re
import sys
import os

# Collections
word_count = Counter(['apple', 'banana', 'apple'])
default_dict = defaultdict(list)
Point = namedtuple('Point', ['x', 'y'])

# Date and time
now = datetime.now()
one_day = timedelta(days=1)
tomorrow = now + one_day

# Regular expressions
pattern = r'\b\w+@\w+\.\w+\b'
email = re.search(pattern, "Contact: user@example.com")

# Path operations
current_dir = Path('.')
file_path = current_dir / 'data' / 'file.txt'


#THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)