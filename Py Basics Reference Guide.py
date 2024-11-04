# PYTHON BASICS REFERENCE GUIDE

#THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)


# 1. VARIABLES AND DATA TYPES
name = "John"          # String - text enclosed in quotes
age = 25              # Integer - whole number
height = 1.75         # Float - decimal number
is_student = True     # Boolean - True or False
none_value = None     # NoneType - represents nothing/null

# 2. BASIC OPERATIONS
# Arithmetic
x = 10 + 5           # Addition
y = 10 - 5           # Subtraction
z = 10 * 5           # Multiplication
a = 10 / 5           # Division (returns float)
b = 10 // 3          # Floor division (returns integer)
c = 10 % 3           # Modulus (remainder)
d = 2 ** 3           # Exponentiation (2 to the power of 3)

# String operations
first = "Hello"
last = "World"
full = first + " " + last    # String concatenation
greeting = f"{first} {last}" # f-strings (formatted strings)
upper_text = "hello".upper() # String methods
lower_text = "HELLO".lower()

# 3. DATA STRUCTURES
# Lists - ordered, mutable collection
my_list = [1, 2, 3, 4, 5]
my_list.append(6)           # Add item to end
my_list.pop()              # Remove last item
my_list[0] = 10            # Modify item

# Tuples - ordered, immutable collection
my_tuple = (1, 2, 3)       # Can't be modified after creation

# Dictionaries - key-value pairs
person = {
    "name": "John",
    "age": 25,
    "city": "New York"
}
person["age"] = 26         # Modify value
person["email"] = "john@example.com"  # Add new key-value pair

# Sets - unordered collection of unique items
my_set = {1, 2, 3, 3}      # Duplicates are removed automatically

# 4. CONTROL FLOW
# If statements
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Loops
# For loop
for i in range(5):         # Loop through range
    print(i)               # Prints 0, 1, 2, 3, 4

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# 5. FUNCTIONS
def greet(name, greeting="Hello"):    # Function with default parameter
    """
    This is a docstring - used to document the function
    """
    return f"{greeting}, {name}!"

# Function call
message = greet("John")
message_custom = greet("John", "Hi")

# 6. LIST COMPREHENSION
numbers = [1, 2, 3, 4, 5]
squares = [n**2 for n in numbers]     # Creates: [1, 4, 9, 16, 25]

# 7. ERROR HANDLING
try:
    result = 10 / 0                   # This will raise an error
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This always executes")

# 8. CLASSES AND OBJECTS
class Dog:
    def __init__(self, name):         # Constructor method
        self.name = name              # Instance variable
    
    def bark(self):                   # Instance method
        return f"{self.name} says Woof!"

# Creating object
my_dog = Dog("Rex")
bark_sound = my_dog.bark()            # Calling method

# 9. MODULES AND IMPORTS
# In file1.py
import math                           # Import entire module
from random import randint           # Import specific function
import pandas as pd                  # Common convention for aliases

# 10. FILE OPERATIONS
# Reading file
with open('file.txt', 'r') as file:  # 'r' for read mode
    content = file.read()

# Writing file
with open('file.txt', 'w') as file:  # 'w' for write mode
    file.write("Hello World")

# 11. LAMBDA FUNCTIONS (Anonymous functions)
square = lambda x: x**2              # Simple one-line function
result = square(5)                   # Returns 25

# 12. COMMON BUILT-IN FUNCTIONS
length = len([1, 2, 3])             # Get length of sequence
maximum = max([1, 2, 3])            # Find maximum value
minimum = min([1, 2, 3])            # Find minimum value
total = sum([1, 2, 3])              # Sum of all values



#THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)