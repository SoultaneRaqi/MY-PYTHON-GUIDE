# ULTIMATE PYTHON PROGRAMMING GUIDE
# Part 1: Advanced Language Features


# THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)




# 1. ADVANCED STRING OPERATIONS
# F-strings with advanced formatting
import decimal
price = decimal.Decimal('1234.5678')
formatted = f"""
Different format options:
Regular:     {price}
Currency:    {price:,.2f}
Scientific:  {price:e}
Percentage:  {price:.2%}
Right align: {price:>10.2f}
Left align:  {price:<10.2f}
Center:      {price:^10.2f}
With sign:   {price:+.2f}
"""

# Advanced string operations
text = "Python Programming"
# String methods chaining
processed = text.lower().replace(' ', '_').center(30, '*')
# Advanced splitting
import re
words = re.split(r'[\s,;]+', "python; java, javascript")

# 2. ADVANCED COLLECTIONS
from collections import (
    ChainMap,
    Counter,
    defaultdict,
    deque,
    namedtuple,
    OrderedDict
)

# ChainMap for multiple dictionaries
defaults = {'theme': 'dark', 'language': 'en'}
user_settings = {'theme': 'light'}
settings = ChainMap(user_settings, defaults)

# Deque for efficient queue operations
queue = deque(maxlen=3)
queue.extend([1, 2, 3, 4])  # Automatically removes oldest items

# Counter with math operations
c1 = Counter(['a', 'b', 'b', 'c'])
c2 = Counter(['b', 'b', 'c', 'd'])
print(c1 & c2)  # Intersection
print(c1 | c2)  # Union

# 3. ADVANCED COMPREHENSIONS
# Nested comprehensions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]

# Set and dict comprehensions with conditions
words = ['hello', 'world', 'python', 'programming']
word_lengths = {
    word: len(word) 
    for word in words 
    if len(word) > 5
}

# Generator expressions with pipe operations
def process_items():
    numbers = (x for x in range(1000))  # Generator expression
    filtered = (x for x in numbers if x % 2 == 0)
    squared = (x**2 for x in filtered)
    return sum(squared)  # Evaluates generators lazily

# 4. DESCRIPTORS AND PROPERTIES
class ValidString:
    def __init__(self, minlen=0, maxlen=None):
        self.minlen = minlen
        self.maxlen = maxlen
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, '')
    
    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise TypeError('Value must be a string')
        if len(value) < self.minlen:
            raise ValueError(f'String must be at least {self.minlen} chars')
        if self.maxlen and len(value) > self.maxlen:
            raise ValueError(f'String must be at most {self.maxlen} chars')
        instance.__dict__[self.name] = value
    
    def __set_name__(self, owner, name):
        self.name = name

# 5. METACLASSES AND CLASS DECORATORS
# Metaclass example
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# Class decorator example
def dataclass_like(cls):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        items = [f"{k}={v!r}" for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(items)})"
    
    cls.__init__ = __init__
    cls.__repr__ = __repr__
    return cls

# Part 2: Design Patterns

# 1. CREATIONAL PATTERNS
# Singleton using metaclass
class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected"

# Factory Method
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError("Invalid animal type")

# 2. STRUCTURAL PATTERNS
# Adapter Pattern
class LegacyAPI:
    def old_method(self):
        return "Legacy Data"

class ModernAPI:
    def new_method(self):
        self.legacy = LegacyAPI()
        return f"Modern: {self.legacy.old_method()}"

# Decorator Pattern
class Coffee:
    def cost(self):
        return 5

class MilkDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost() + 2

# 3. BEHAVIORAL PATTERNS
# Observer Pattern
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)

# Part 3: Advanced Algorithms and Data Structures

# 1. CUSTOM DATA STRUCTURES
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
            return
        
        def _insert(node, value):
            if value < node.value:
                if node.left is None:
                    node.left = TreeNode(value)
                else:
                    _insert(node.left, value)
            else:
                if node.right is None:
                    node.right = TreeNode(value)
                else:
                    _insert(node.right, value)
        
        _insert(self.root, value)

# 2. SORTING ALGORITHMS
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Part 4: Advanced Python Techniques

# 1. MEMORY OPTIMIZATION
import sys
from typing import NamedTuple

# Using __slots__
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Using NamedTuple instead of class
class PointTuple(NamedTuple):
    x: float
    y: float

# 2. PERFORMANCE OPTIMIZATION
from functools import lru_cache
import time

# Memoization with LRU Cache
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Context manager for timing
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

# 3. CONCURRENT PROGRAMMING
import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Async/await pattern
async def fetch_data(url):
    await asyncio.sleep(1)  # Simulate IO
    return f"Data from {url}"

async def process_urls(urls):
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Thread pool
def cpu_bound(n):
    return sum(i * i for i in range(n))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_bound, [1000000] * 4))

# Process pool
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_bound, [1000000] * 4))

# 4. NETWORKING AND WEB
import socket
import requests
import aiohttp
import websockets

# Basic socket server
def create_server(host='localhost', port=8888):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    return server

# Async web client
async def fetch_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(session.get(url))
        responses = await asyncio.gather(*tasks)
        return responses

# WebSocket handler
async def websocket_handler(websocket, path):
    async for message in websocket:
        await websocket.send(f"Echo: {message}")

# 5. SECURITY AND ENCRYPTION
import hashlib
import secrets
from cryptography.fernet import Fernet

# Secure password hashing
def hash_password(password: str) -> tuple:
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return salt, key

# Symmetric encryption
def encrypt_message(message: str) -> tuple:
    key = Fernet.generate_key()
    f = Fernet(key)
    encrypted = f.encrypt(message.encode())
    return key, encrypted

# Part 5: Testing and Quality Assurance

# 1. UNIT TESTING
import unittest
from unittest.mock import Mock, patch

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_addition(self):
        self.assertEqual(self.calc.add(2, 2), 4)
    
    @patch('module.external_service')
    def test_with_mock(self, mock_service):
        mock_service.return_value = 42
        result = self.calc.process()
        self.assertEqual(result, 42)

# 2. PROPERTY TESTING
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_idempotent(lst):
    sorted_once = sorted(lst)
    sorted_twice = sorted(sorted_once)
    assert sorted_once == sorted_twice

# 3. TYPE CHECKING
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

def render(drawable: Drawable) -> None:
    drawable.draw()

# Part 6: Best Practices and Project Structure

# 1. PROJECT STRUCTURE
"""
my_project/
├── src/
│   └── package/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
├── docs/
├── requirements.txt
├── setup.py
└── README.md
"""

# 2. CONFIGURATION MANAGEMENT
import configparser
import yaml
import json
from pathlib import Path

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

# 3. LOGGING AND MONITORING
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger(__name__)
    handler = RotatingFileHandler(
        'app.log',
        maxBytes=1024*1024,
        backupCount=5
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# 4. DOCUMENTATION
def example_function(param1: str, param2: int) -> bool:
    """
    Example function with Google-style docstring.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
    
    Returns:
        True if successful, False otherwise.
    
    Raises:
        ValueError: If param2 is negative.
        
    Examples:
        >>> example_function("test", 42)
        True
    """
    if param2 < 0:
        raise ValueError("param2 must be non-negative")
    return True
  
  
  # THAT WAS MADE BY SOULTANE RAQI (https://github.com/raqisoultane)