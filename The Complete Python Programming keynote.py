###############################################################################
#                    THE COMPLETE PYTHON PROGRAMMING KEYNOTE                     #
#                  A Comprehensive Guide with Full Comments                     #
###############################################################################
# THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)
####################
# SECTION 1: FUNDAMENTALS
####################

# 1.1 BASIC DATA TYPES AND VARIABLES
#----------------------------------

# Numbers
integer_num = 42        # Whole number
float_num = 3.14       # Decimal number
complex_num = 1 + 2j   # Complex number (real + imaginary)
hex_num = 0xFF        # Hexadecimal number (base 16) - equals 255
oct_num = 0o777       # Octal number (base 8) - equals 511
bin_num = 0b1010      # Binary number (base 2) - equals 10

# Basic arithmetic operations
addition = 5 + 3       # 8
subtraction = 5 - 3    # 2
multiplication = 5 * 3 # 15
division = 5 / 3       # 1.6666... (float division)
floor_div = 5 // 3     # 1 (integer division, rounds down)
modulus = 5 % 3        # 2 (remainder)
power = 5 ** 3         # 125 (5 to the power of 3)

# Strings - Text data
single_quote = 'Hello'              # String with single quotes
double_quote = "World"              # String with double quotes
multi_line = '''This is a          
multi-line string'''                # Multi-line string
raw_string = r"C:\new\folder"       # Raw string (backslashes not escaped)

# String operations
name = "John Doe"
print(len(name))          # 8 (string length)
print(name.upper())       # "JOHN DOE" (all uppercase)
print(name.lower())       # "john doe" (all lowercase)
print(name.split())       # ["John", "Doe"] (split into list)
print(name.replace('o', 'x'))  # "Jxhn Dxe" (replace characters)
print(name[0:4])          # "John" (slicing - start:end)
print(name[-3:])          # "Doe" (negative indexing)

# F-strings (formatted strings) - Python 3.6+
age = 30
height = 1.75
message = f"Name: {name}, Age: {age}, Height: {height:.2f}m"

# Boolean - True/False
is_active = True
is_empty = False
result = None            # None type - represents absence of value

# Type conversion
str_num = "123"
num = int(str_num)       # String to integer
str_float = "3.14"
float_num = float(str_float)  # String to float
str_result = str(123)    # Number to string

# 1.2 COLLECTIONS AND DATA STRUCTURES
#----------------------------------

# Lists - Ordered, mutable collection
#----------------------------------
# Creating lists
fruits = ['apple', 'banana', 'orange']  # Basic list
mixed_list = [1, 'hello', 3.14, True]   # List with mixed types
nested_list = [1, [2, 3], [4, 5, 6]]    # Nested list

# List operations
fruits.append('grape')     # Add item to end
fruits.insert(1, 'mango') # Insert at position
fruits.remove('banana')    # Remove first occurrence
popped = fruits.pop()      # Remove and return last item
fruits.sort()              # Sort in place
fruits.reverse()           # Reverse in place
fruits.extend(['kiwi', 'pear'])  # Add multiple items

# List comprehensions
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]        # [1, 4, 9, 16, 25]
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16]

# Tuples - Ordered, immutable collection
#-------------------------------------
# Creating tuples
coordinates = (10, 20)     # Basic tuple
single_tuple = (1,)        # Single item tuple (note the comma)
nested_tuple = (1, (2, 3), 4)  # Nested tuple

# Tuple operations
x, y = coordinates        # Tuple unpacking
combined = coordinates + (30,)  # Creating new tuple

# Dictionaries - Key-value pairs
#-----------------------------
# Creating dictionaries
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}

# Dictionary operations
person['email'] = 'john@example.com'  # Add/update item
del person['age']                     # Remove item
keys = person.keys()                  # Get keys view
values = person.values()              # Get values view
items = person.items()                # Get key-value pairs

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}

# Sets - Unordered collection of unique items
#-----------------------------------------
# Creating sets
numbers_set = {1, 2, 3, 4, 5}
words_set = set(['apple', 'banana', 'apple'])  # Duplicates removed

# Set operations
numbers_set.add(6)        # Add item
numbers_set.remove(1)     # Remove item (raises error if missing)
numbers_set.discard(1)    # Remove item (no error if missing)
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print(set1 | set2)        # Union {1, 2, 3, 4, 5}
print(set1 & set2)        # Intersection {3}
print(set1 - set2)        # Difference {1, 2}

####################
# SECTION 2: CONTROL FLOW
####################

# 2.1 CONDITIONALS
#---------------

# If statements
age = 18
if age < 13:
    print("Child")
elif age < 20:
    print("Teenager")
else:
    print("Adult")

# Ternary operator (one-line if-else)
status = "adult" if age >= 18 else "minor"

# Match statement (Python 3.10+)
command = "quit"
match command:
    case "quit":
        print("Exiting...")
    case "restart":
        print("Restarting...")
    case _:                # Default case
        print("Unknown command")

# 2.2 LOOPS
#---------

# For loops
for i in range(5):        # Loop 5 times
    print(i)              # Prints 0 to 4

# Loop with enumerate
for index, value in enumerate(['a', 'b', 'c']):
    print(f"{index}: {value}")

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control
for i in range(10):
    if i == 3:
        continue        # Skip rest of current iteration
    if i == 8:
        break          # Exit loop
    print(i)

# Loop with else
for i in range(5):
    print(i)
else:                  # Executed when loop completes normally
    print("Loop completed")

####################
# SECTION 3: FUNCTIONS AND MODULES
####################

# 3.1 FUNCTION BASICS
#------------------

# Basic function
def greet(name):
    """
    Simple greeting function.
    Args:
        name: Person's name
    Returns:
        Greeting string
    """
    return f"Hello, {name}!"

# Function with default parameters
def power(base, exponent=2):
    return base ** exponent

# Function with *args (variable positional arguments)
def sum_all(*args):
    return sum(args)

# Function with **kwargs (variable keyword arguments)
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Lambda functions (anonymous functions)
square = lambda x: x**2
product = lambda x, y: x * y

# 3.2 ADVANCED FUNCTION CONCEPTS
#----------------------------

# Closure (function factory)
def make_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

# Decorator
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

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 3.3 MODULES AND PACKAGES
#----------------------

# Importing modules
import math
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np  # Common convention for numpy

# Creating modules
# In file: mymodule.py
def my_function():
    return "Hello from my module!"

# In another file:
import mymodule
mymodule.my_function()

####################
# SECTION 4: OBJECT-ORIENTED PROGRAMMING
####################

# 4.1 CLASSES AND OBJECTS
#----------------------

# Basic class
class Dog:
    # Class variable (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor
    def __init__(self, name, age):
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    # String representation
    def __str__(self):
        return f"{self.name} is {self.age} years old"
    
    # Representation for developers
    def __repr__(self):
        return f"Dog(name='{self.name}', age={self.age})"

# Creating objects
dog1 = Dog("Rex", 3)
dog2 = Dog("Buddy", 5)

# 4.2 INHERITANCE AND POLYMORPHISM
#------------------------------

# Parent class
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement")

# Child classes
class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

class Duck(Animal):
    def speak(self):
        return f"{self.name} says Quack!"

# 4.3 SPECIAL METHODS
#-----------------

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):        # Addition
        return Point(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):         # Equality
        return self.x == other.x and self.y == other.y
    
    def __len__(self):              # Length
        return int((self.x ** 2 + self.y ** 2) ** 0.5)

####################
# SECTION 5: ADVANCED CONCEPTS
####################

# 5.1 EXCEPTION HANDLING
#--------------------

# Basic exception handling
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except (TypeError, ValueError) as e:
    print(f"Error: {e}")
else:
    print("No error occurred")
finally:
    print("This always executes")

# Custom exception
class CustomError(Exception):
    pass

# 5.2 CONTEXT MANAGERS
#------------------

# Using with statement
with open('file.txt', 'w') as f:
    f.write('Hello World')

# Custom context manager
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"Elapsed: {self.end - self.start}")

# 5.3 ADVANCED CONCEPTS
#-------------------

# Type hints (Python 3.5+)
from typing import List, Dict, Optional

def process_items(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

# Async/await (Python 3.5+)
import asyncio

async def main():
    print('Hello')
    await asyncio.sleep(1)
    print('World')

# Dataclasses (Python 3.7+)
from dataclasses import dataclass

@dataclass
class Point3D:
    x: float
    y: float
    z: float

####################
# SECTION 6: PRACTICAL EXAMPLES
####################

# 6.1 FILE HANDLING
#---------------

# Reading files
with open('file.txt', 'r') as f:
    content = f.read()         # Read entire file
    lines = f.readlines()      # Read lines into list

# Writing files
with open('output.txt', 'w') as f:
    f.write('Hello\n')
    f.writelines(['Line 1\n', 'Line 2\n'])

# 6.2 JSON HANDLING
#---------------

import json

# Writing JSON
data = {'name': 'John', 'age': 30}
json_string = json.dumps(data, indent=2)

# Reading JSON
parsed_data = json.loads(json_string)

# 6.3 DATABASE ACCESS
#-----------------

import sqlite3

# Connect to database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users
    (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)
''')

# Insert data
cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)',
              ('John', 30))

# Query data
cursor.execute('SELECT * FROM users')
results = cursor.fetchall()

# 6.4 WEB REQUESTS
#--------------

import requests

# GET request
response = requests.get('https://api.example.com/data')
data = response.json()

# POST request
response = requests.post(
    'https://api.example.com/submit',
    json={'key': 'value'}
)

####################
# SECTION 7: TESTING AND DEBUGGING
####################

# 7.1 UNIT TESTING
#--------------

import unittest

class TestMathFunctions(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)
    
    def test_division(self):
        with self.assertRaises(ZeroDivisionError):
            1 / 0

# 7.2 DEBUGGING
#-----------

# Using pdb
import pdb

def complex_function():
    x = 5
    y = 0
    pdb.set_trace()  # Debugger breakpoint
    z = x / y
    return z

# 7.3 LOGGING
#----------

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Using logger
logger = logging.getLogger(__name__)
logger.info("Application started")
####################
# SECTION 8: PERFORMANCE AND OPTIMIZATION 
####################

# 8.1 PROFILING
#------------

import cProfile
import profile

# Using cProfile
def profile_function():
    cProfile.run('my_function()')

# Line profiler
@profile
def slow_function():
    result = 0
    for i in range(1000000):
        result += i
    return result

# 8.2 MEMORY OPTIMIZATION
#---------------------

# Generator expressions (memory efficient)
sum(x * x for x in range(1000000))

# Using slots to reduce memory usage
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 8.3 CODE OPTIMIZATION
#-------------------

# List comprehension vs loop
# Faster: [x**2 for x in range(1000)]
# Slower: 
squares = []
for x in range(1000):
    squares.append(x**2)

# Using built-in functions
# Faster: sum(numbers)
# Slower:
total = 0
for num in numbers:
    total += num

####################
# SECTION 9: ADVANCED PYTHON FEATURES
####################

# 9.1 METACLASSES
#-------------

# Custom metaclass
class MetaClass(type):
    def __new__(cls, name, bases, attrs):
        # Add methods or attributes to class
        attrs['custom_attribute'] = 'Added by metaclass'
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=MetaClass):
    pass

# 9.2 DESCRIPTORS
#-------------

class Descriptor:
    def __get__(self, obj, owner=None):
        return 42
    
    def __set__(self, obj, value):
        raise AttributeError("Cannot modify")

class MyClass:
    x = Descriptor()

# 9.3 COROUTINES
#------------

import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {'data': 'example'}

async def process_data():
    data = await fetch_data()
    return data

####################
# SECTION 10: BEST PRACTICES AND PATTERNS
####################

# 10.1 DESIGN PATTERNS
#------------------

# Singleton pattern
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Factory pattern
class AnimalFactory:
    def create_animal(self, animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()

# 10.2 CODE STYLE
#-------------

# Following PEP 8
def calculate_average(numbers: list) -> float:
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        float: The average value
    """
    return sum(numbers) / len(numbers)

# 10.3 PROJECT STRUCTURE
#-------------------

"""
typical_project/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── tests/
│   ├── __init__.py
│   └── test_main.py
│
├── docs/
│   └── README.md
│
├── requirements.txt
└── setup.py
"""

####################
# SECTION 11: CONCURRENCY AND PARALLELISM
####################

# 11.1 THREADING
#------------

import threading

def worker():
    print(f"Thread {threading.current_thread().name}")

# Create and start thread
thread = threading.Thread(target=worker)
thread.start()
thread.join()

# Thread synchronization
lock = threading.Lock()
with lock:
    # Critical section
    pass

# 11.2 MULTIPROCESSING
#------------------

from multiprocessing import Process, Pool

def process_worker(x):
    return x * x

# Using Process
p = Process(target=process_worker, args=(10,))
p.start()
p.join()

# Using Pool
with Pool(4) as pool:
    result = pool.map(process_worker, range(10))

# 11.3 ASYNC IO
#-----------

import asyncio

async def async_worker():
    await asyncio.sleep(1)
    return "Done"

# Run async function
asyncio.run(async_worker())

####################
# SECTION 12: SECURITY AND DEPLOYMENT
####################

# 12.1 SECURITY BEST PRACTICES
#--------------------------

# Secure password handling
import hashlib
import secrets

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest()

# Input validation
def validate_input(user_input: str) -> bool:
    return user_input.isalnum() and len(user_input) < 100

# 12.2 ENVIRONMENT VARIABLES
#-----------------------

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')

# 12.3 LOGGING SECURITY EVENTS
#--------------------------

import logging

logging.basicConfig(
    filename='security.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

####################
# END OF PYTHON PROGRAMMING KEYNOTE
####################
# THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)