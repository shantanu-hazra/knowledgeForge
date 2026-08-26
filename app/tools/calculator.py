def add(*args: float) -> float:
    return sum(args)

def mult(*args: float) -> float:
    result = 1
    for n in args:
        result *= n
    return result

def sub(a: float, b: float) -> float:
    return a - b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def pct(a: float, b: float) -> float:
    return (a * 100) / b