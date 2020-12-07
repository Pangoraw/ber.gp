---
title: "Converting a python Dict into a class"
date: 2020-12-07T11:05:33+01:00
draft: false
tags:
  - Python
---

In python, dictionnaries and classes behave differently. One of the main difference is how you access their members. In the following example, `class_instance` and `dictionnary` hold the same data.

```python
class Foo:
  def __init__(self):
    self.bar = "foo"

class_instance = Foo()
dictionnary = {'bar': 'foo'}

print(class_instance.bar, dictionnary['bar']) # foo foo
```

But in order to access the value of `bar`, the syntax is longer for the dictionnary and may not be convenient in cases where the class syntax is required. To quickly convert the dictionnary into a class, you can use the following code:

```python
new_class = type('D', (object,), dictionnary)()
print(new_class.bar) # foo
```

Under the hood, it will construct a class called `D` and build it using the dictionnary's key/value pairs.
