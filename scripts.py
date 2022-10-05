# Introduction - Hello world
if __name__ == '__main__':
    print("Hello, World!")

# Introduction - Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
    
    if n%2 != 0:
        print("Weird")
    elif n in range(2,6):
        print("Not Weird")
    elif n in range(6,21):
        print("Weird")
    else:
        print("Not Weird")

# Introduction - Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a+b)
    print(a-b)
    print(a*b)

# Introduction - Python Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    print(a//b,"\n",a/b)

# Introduction - Loops
if __name__ == '__main__':
    n = int(input())
    
    for i in range(n):
        print(i**2)

# Introduction - Write a function
def is_leap(year):
    leap = False
    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True
    return leap

year = int(input())
print(is_leap(year))

# Introduction - Print Function
if __name__ == '__main__':
    n = int(input())
    s = ""
    for i in range(1,n+1):
        s += str(i)
    print(s)

# Basic Data Types - List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    

    res = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k !=n]
    print(res)

# Basic Data Types - Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    l = list(arr)
    m = max(l)
    l.sort()
    while m in l:
        l.remove(m)
    print(l[-1])

# Basic Data Types - Nested Lists
def  score_getter(l:list):
    return l[1]


if __name__ == '__main__':
    l = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        l.append([name,score])      # creating student nested list
    l.sort(key=score_getter)        # sorting on the score
    
    m = l[0][1]         # getting the min
    
    second_lowest = [s for s in l if s[1]>m]    # removing the lowest scores
    
    new_m = second_lowest[0][1]
    res = []
    for s in second_lowest:
        if s[1] == new_m:
            res.append(s[0])
    
    res.sort()      # alphabetical order
    for name in res:
        print(name)


# Basic Data Types - Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
    perc = str(int(sum(student_marks[query_name])/3*100))
    print(f"{perc[:-2]}.{perc[-2:]}")

# Basic Data Types - Lists
if __name__ == '__main__':
    N = int(input())
    l,commands = [],[]
    for _ in range(N):
        c = input()
        commands.append(c)
    
    
    for c in commands:
        #print(l)
        if "insert" in c:
            pos, i = c.split(" ")[1],c.split(" ")[2]
            l.insert(int(pos),int(i))
        elif c == "print":
            print(l)
        elif "remove" in c:
            l.remove(int(c.split(" ")[1]))
        elif "append" in c:
            l.append(int(c.split(" ")[1]))
        elif c == "sort":
            l.sort()
        elif c == "pop":
            l.pop(-1)
        elif c == "reverse":
            l.reverse()

# Basic Data Types - Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    
    print(hash(tuple(integer_list)))

# Strings - sWAP cASE
def swap_case(string):
    l = [s.upper() if s.islower() else s.lower() for s in string]   # collect all swapped char into a list
    return "".join(l)   # back to a string

# Strings - String Split and Join
def split_and_join(line):
    # write your code here
    l = line.split(" ")
    return "-".join(l)
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# Strings - What's Your Name?
def print_full_name(first, last):
    # Write your code here
    print(f"Hello {first} {last}! You just delved into python.")

# Strings - Mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

# Strings - Find a string
def count_substring(string, sub_string):
    l = [string[i:i+len(sub_string)] for i in range(len(string))]
    
    return l.count(sub_string)

# Strings - String Validators
if __name__ == '__main__':
    num, alpha, digit, lower, upper = False,False,False,False,False
    string = input()
    for s in string:
        if s.isalnum():
            num = True
        if s.isalpha():
            alpha = True
        if s.islower():
            lower = True
        if s.isdigit():
            digit = True 
        if s.isupper():
            upper = True
    print(num)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)

# Strings - Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Strings - Text Wrap
def wrap(string, max_width):
    s = ""
    n = len(string)//max_width  # number of sub_strings
    for i in range(n):
        s += string[i*max_width:(i+1)*max_width] + "\n"
    s += string[n*max_width:]    # for the left out characters
    return s

# Strings - Designer Door Mat
inp = input().split(" ")
n,m = int(inp[0]),int(inp[1])

center = "WELCOME".center(m,"-")

lines,base = [], ".|."
for i in range(n//2):
    s = base*i + base+ base*i
    lines.append(s.center(m,"-"))

for line in lines:
    print(line)
print(center)
lines.reverse()
for line in lines:
    print(line)

# Strings - Capitalize!
def solve(s):
    fin = []
    words = s.split(" ") 
    for w in words:
        fin.append(w.capitalize())
    return " ".join(fin)

# Strings - Alphabet Rangoli
def list_spec(s):   
    """invert and merge a string"""
    l = list(s)
    l.reverse()
    return "".join(l[:-1]) + s

def print_rangoli(size):
    letters = [chr(97+i) for i in range(size)]  # all letters used
        
    bottom_right = ["-".join(letters[i:]).ljust(size*2-1,"-") for i in range(size)]
    bottom = list(map(list_spec,bottom_right))  # bottom half
    top = bottom[1:].copy() # reverse bottom half for top half
    top.reverse()
    
    for r in top:
        print(r)
    for r in bottom:
        print(r)

# Strings - The Minion Game
# It is not correct but I don't understand the "correct" answers on hackerrank
def count(string,substring):    # builtin str count function not working on overlaps
    substrings = [string[i:i+len(substring)] for i in range(len(string))]
    return substrings.count(substring)

    
def substrings(string,char):
    c = 0
    p = string.find(char)
    if p != -1:
        sub_strings = string[p:]
        for s in range(1,len(string)-p+1):
            sub_string = sub_strings[:s]
            c += count(string,sub_string)
    return c
    
def minion_game(string):
    d = {
        "Stuart":0,
        "Kevin":0
    }
    
    # vowels for Kevin
    for v in ["A","E","I","O","U"]:
        d["Kevin"] += substrings(string,v)

    # consonants for Stuart
    for c in range(ord("B"),ord("Z")):
        if c not in [ord("E"),ord("I"),ord("O"),ord("U"),]:
            d["Stuart"] += substrings(string,chr(c))
    
    if d["Kevin"] >= d["Stuart"]:
        print("Kevin "+str(d["Kevin"]))
    else:
        print("Stuart "+str(d["Stuart"]))
            
# Strings - String Formatting
def print_formatted(number):
    width = len(str(bin(number)).lstrip("0b"))+1 # max len for padding
    for i in range(1,number+1):
        n = str(i)
        b = str(bin(i)).lstrip("0b")
        o = str(oct(i)).lstrip("0o")
        h = str(hex(i)).lstrip("0x").upper()
        
        print(n.rjust(width-1," ")+o.rjust(width," ")+h.rjust(width," ")+b.rjust(width," "))

# Strings - Merge the Tools!
def merge_the_tools(string, k):
    for i in range(len(string)//k):
        s = set()
        fin = ""
        substring = string[i*k:i*k+k]
        for c in substring:
            if c not in s:
                s.add(c)
                fin+=c
        print(fin)

# Sets - Introduction to Sets
def average(array):
    # your code goes here
    return sum(set(array))/len(set(array))

# Sets - Check Subset
test = int(input())
for t in range(test):
    la = input()
    a = set(input().split(" "))
    lb = input()
    b = set(input().split(" "))
    print(a.issubset(b))

# Sets - Set .add() 
n = int(input())
s = set()
for i in range(n):
    s.add(input())

print(len(s))

# Sets - Set .union() Operation
n1 = input()
a = set(input().split(" "))
n2 = input()
b = set(input().split(" "))

print(len(a.union(b)))

# Sets - Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))

for i in range(int(input())):
    comand = input()
    try:
        if "remove" in comand:
            s.remove(int(comand.split(" ")[1]))
        elif "discard" in comand:
            s.discard(int(comand.split(" ")[1]))
        elif "pop" in comand:
            s.pop()
    except Exception:
        continue
print(sum(s))

# Sets - Set .intersection() Operation
input()
a = set(input().split(" "))
input()
b = set(input().split(" "))
print(len(a.intersection(b)))

# Sets - Set .difference() Operation
input()
a = set(input().split(" "))
input()
b = set(input().split(" "))
print(len(a.difference(b)))

# Sets - Set .symmetric_difference() Operation
input()
a = set(input().split(" "))
input()
b = set(input().split(" "))
print(len(a^b))

# Sets - Set Mutations
useless = input()   # why have the len of the set
a = set(input().split(" "))

n = input()
for s in range(int(n)):
    comand = input().split(" ")[0] # the len of the set is useless
    new_set = input().split(" ")
    if comand == "intersection_update":
        a.intersection_update(new_set)
    elif comand == "update":
        a.update(new_set)
    elif comand == "symmetric_difference_update":
        a.symmetric_difference_update(new_set)
    elif comand == "difference_update":
        a.difference_update(new_set)

print(sum(map(int,a)))

# Sets - Symmetric Difference
useless, a = input(), set(input().split(" "))
useless, b = input(), set(input().split(" "))

l = list(map(int,a.symmetric_difference(b)))
l.sort()
for i in l:
    print(i)

# Sets - Check Strict Superset
def sup(a:set,sets:list):
    for s in sets:
        if s == a or not a.issuperset(s):
            return False
    return True

if __name__ == "__main__":
    a,n = set(input().split(" ")),int(input())
    l = []
    for i in range(n):
        l.append(set(input().split(" ")))
    print(sup(a,l))

# Sets - No Idea!
n_m = input().split(" ")
l = input().split(" ")
a,b = set(input().split(" ")), set(input().split(" "))

print(sum([1 if i in a else -1 if i in b else 0 for i in l]))

# Sets - The Captain's Room
n,rooms = int(input()),input().split(" ")
#cap = [r for r in set(rooms) if rooms.count(r) != n]
s = set(rooms)
for i in s:
    if rooms.count(i) == 1:
        print(i)
        break


# Collections - DefaultDict Tutorial
from collections import defaultdict
n_m = input().split(" ")
d = defaultdict(list)
for n in range(int(n_m[0])):
    d["a"].append(input())
for n in range(int(n_m[1])):
    d["b"].append(input())

for i in d["b"]:
    res, a, c = "",d["a"].copy(),1
    if i not in a:
        res = "-1"
    while i in a:
        ind = a.index(i)
        res += str(ind+c) + " "
        a.pop(ind)
        c += 1
    print(res.rstrip(" "))

# Collections - collections.Counter()
from collections import Counter
x = int(input())
sizes = Counter(input().split(" "))
n = int(input())
total = 0

for c in range(n):
    customer = input().split(" ")
    shoe, price = customer[0], int(customer[1])
    sizes[shoe] += -1
    if sizes[shoe] >= 0:
        total += price
print(total)

# Collections - Collections.OrderedDict()
from collections import OrderedDict
d = OrderedDict()
for i in range(int(input())):
    item = input().split()
    name = " ".join(item[:-1])
    if name in d.keys():
        d[" ".join(item[:-1])] += int(item[-1])
    else:
        d[" ".join(item[:-1])] = int(item[-1])
    
for i in d.items():
    print(f"{i[0]} {i[1]}")

# Collections - Collections.deque()
from collections import deque
d = deque()

for i in range(int(input())):
    c = input().split()
    if c[0] == "append":
        d.append(c[1])
    elif c[0] == "appendleft":
        d.appendleft(c[1])
    elif c[0] == "appendleft":
        d.appendleft(c[1])
    elif c[0] == "extend":
        d.extend(c[1])
    elif c[0] == "pop":
        d.pop()
    elif c[0] == "popleft":
        d.popleft()
print(" ".join(list(d)))

# Collections - Word Order
from collections import Counter
n = int(input())
l = []
for i in range(n):
    string = input()
    l.append(string)

d = Counter(l)
print(len(d.values()))
s = ""
for i in d.items():
    s += str(i[1]) + " "
print(s.rstrip(" "))

# Collections - Company Logo
from collections import Counter

if __name__ == '__main__':
    s = input()
    l = list(Counter(s).items()) # create the list of the key-value tuples
    l.sort(key = lambda x:  (-x[1],x[0]))   # sort on the occurences (descending) and in second place on alphabeth
    top = l[:3]

    for i in top:
        print(i[0],i[1])

# Collections - Piling Up!
from collections import deque

def case_check(blocks):
    m = max(blocks[0],blocks[-1])
    while len(blocks)>=1:
        if max(blocks[0],blocks[-1]) > m:   # if the sides are both larger than the cube returns no
            return "No"
        if blocks[0] == max(blocks[0],blocks[-1]):
            blocks.popleft()
        elif blocks[-1] == max(blocks[0],blocks[-1]):
            blocks.pop() 
    return "Yes"    # if  the cycle ends it means that the cube can be stacked

if __name__ == "__main__":
    cases = int(input())
    for t in range(cases):
        _,blocks = input(), deque(map(int,input().split()))
        print(case_check(blocks))

# Collections - Collections.namedtuple()
from collections import namedtuple

n, cols, tot  = int(input()),input(),0
Stud = namedtuple("Stud",",".join(cols.split())) # getting columns name
for i in range(n):
    inp = input().split()
    arg1, arg2, arg3, arg4 = inp[0],inp[1],inp[2],inp[3] # passing args in order
    stud = Stud(arg1, arg2, arg3, arg4)
    tot += int(stud.MARKS)
print(tot/n)

# Date and Time - Calendar Module
import calendar

m_d_y = list(map(int,input().split()))
n = calendar.weekday(m_d_y[2],m_d_y[0],m_d_y[1])
print(calendar.day_name[n].upper())

# Date and Time - Time Delta
# Wrong Answer

# Exceptions
for cases in range(int(input())):
    try:
        i = list(map(int,input().split()))
        a,b = i[0], i[1]
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

# Built-ins - Zipped!
l = []
for line in range(int(input().split()[1])):
    marks = list(map(float,input().split()))
    l.append(marks)

for student in zip(*l):
    print(sum(student)/len(student))

# Built-ins - ginortS
s = input()

# dividing the 3 categories
low = [i for i in s if i.islower()]
up = [i for i in s if i.isupper()]
digit = list(map(int,[i for i in s if i.isdigit()]))    # converting digits to type int

low.sort()
up.sort()
digit.sort(key = lambda x : (x%2==0,x)) # sorting odds before even
print("".join(low) + "".join(up) + "".join(list(map(str,digit))))

# Built-ins - Athlete Sort
if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
        
    k = int(input())
    
    arr.sort(key = lambda x: x[k]) # sorting on the k-th column
    for line in arr:
        res = list(map(str,line)) # coverting to str
        print(" ".join(res))

# Python Functionals - Map and Lambda Function
cube = lambda x: x**3

def fibonacci_numbers(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci_numbers(n-1) + fibonacci_numbers(n-2)

def fibonacci(n):
    # return a list of fibonacci numbers
    l = []
    for i in range(n):
        l.append(fibonacci_numbers(i))
    return l

# XML - XML 1 - Find the Score
def get_attr_number(node):
    points = 0
    # referencing the offical documentation "https://docs.python.org/3/library/xml.etree.elementtree.html"

    for el in node.iter():
        points += len(el.attrib)
    return points

# XML - XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    i = 1
    if len(elem.findall("./")) == 0:
        maxdepth = 0
        return
    while elem is not None:
        for child in elem:
            if len(child.findall("./"))>0:    # checks if child has childs (path = "./")
                elem = child
                i += 1
                break
            elem = None
    maxdepth = i

# Closures and Decorators - Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        fin = []
        for s in l:
            mid = s
            if len(s) != 10:
                if s[0] == "0":
                    mid = s.lstrip("0")
                else:
                    mid = s.lstrip("+").lstrip("9").lstrip("1")
            new = "+91 "+ mid[:5] + " "+mid[5:]
            fin.append(new)
        f(fin)
    return fun

# Closures and Decorators - Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        people.sort(key=lambda x : int(x[2]))
        for pearson in people:
            yield f(pearson)
    return inner

# Regex and Parsing - Re.split()
regex_pattern = r"[.,]"

# Regex and Parsing - Detect Floating Point Number
def checker(n:str):
    if len(n.split(".")) == 1:
        return False
    if len(n.split(".")) > 2:
        return False
    try:
        a = float(n)
        return True
    except Exception:
        return False
            
for _ in range(int(input())):
    n = input()
    print(checker(n))

# Regex and Parsing - Group(), Groups() & Groupdict()
import re

s = input()

r = r'([^_])\1'

m = re.search(r,s)
if m is None :
    print(-1)
else:
    print(m.group()[0][0])

# Regex and Parsing - Re.findall() & Re.finditer()
import re
s = input()
r = r"[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]{0,1}[aeiouAEIOU]{2,}[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]"

consonants = "[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]"

l = re.findall(r,s)
if len(l) == 0:
    print(-1)
else:
    for i in l:
        out = re.sub(consonants,"",i)
        print(out)

# Regex and Parsing - Re.start() & Re.end()
import re

s,sub = input(), input()

m = re.finditer(r"(?="+sub+")",s)
not_found = True

for i in m:
    not_found = False   
    print((i.start(),i.start() + len(sub)-1))

if not_found:
    print((-1,-1))

# Regex and Parsing - Validating phone numbers
import re

def validating(s):
    m = re.search(r"^\d{10}$",s)
    if m is None:
        print("NO")
        return
    m = re.match(r"^[789]{1}",s)
    if m is None:
        print("NO")
        return
    print("YES")
    return

for _ in range(int(input())):
    s = input()
    validating(s)


# Regex and Parsing - Validating and Parsing Email Addresses
import email.utils
import re

for _ in range(int(input())):
    l = input()
    s = email.utils.parseaddr(l)[1]
    r = r"^[a-z]+[\w\.\d-]*@[a-zA-Z]+\.[a-z]{1,3}$"
    #print(re.findall(r,s))
    
    m = re.search(r,s)
    if m is not None:
        print(l)

# Regex and Parsing - Hex Color Code
import re

r = r"#[0-9a-fA-F]{3,6}"
res = []
for _ in range(int(input())):
    i = input()
    if ";" in i:
        res += re.findall(r,i)

for match in res:
    print(match)


# Regex and Parsing - Validating Credit Card Numbers
import re 

p = r"^[456]\d{15}|(^[456]\d{3}-)(\d{4}-){2}\d{4}"

for _ in range(int(input())):
    s = input()
    #print(s)
    #print(re.fullmatch(p,s))
    no = s.replace("-","",s.count("-"))
    if len(re.findall(r"(\d)\1{3}",no))> 0 :
        print("Invalid")
        continue
    if re.fullmatch(p,s) is not None:
        print("Valid")
    else:
        print("Invalid")
        

# Regex and Parsing - Matrix Script
first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

res = ""

for i in range(m):
    for w in matrix:
        res += w[i]
print(re.sub(r"(?<=[a-zA-Z0-9])[!@#$%& ]+(?=[a-zA-Z0-9])"," ",res))


# Regex and Parsing - Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser # As for the others html problems 

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        if len(attrs) > 0:
            for c in attrs:
                print(f"-> {c[0]} > {c[1]}")
            
    def handle_comment(self, data):
        """ignore commments"""
        ... 

f = ""
for _ in range(int(input())):
    f += input()
    
parser = MyHTMLParser()
parser.feed(f)


# Regex and Parsing - HTML Parser - Part 1
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for c in attrs:
            print(f"-> {c[0]} > {c[1]}")

    def handle_endtag(self, tag):
        print("End   :", tag)

    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for c in attrs:
            print(f"-> {c[0]} > {c[1]}")

f=""
for _ in range(int(input())):
    f += input()
    
parser = MyHTMLParser()
parser.feed(f)

# Regex and Parsing - HTML Parser - Part 2
from html.parser import HTMLParser 

class MyHTMLParser(HTMLParser):
    def handle_comment(self,data):
        if "\n" in data:
            print(">>> Multi-line Comment")
            print(data.rstrip("\n"))
        else:
            print(">>> Single-line Comment")
            print(data)
    
    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            print(data.replace("\n","",data.count("\n")))
            
f=""
for _ in range(int(input())):
    f += input() + "\n"
    
parser = MyHTMLParser()
parser.feed(f)

# Regex and Parsing - Validating UID
import re

for _ in range(int(input())):
    s = input()
    if re.match(r"([\d\w])*.*\1",s) is not None:
        print("Invalid")
        continue
    if re.fullmatch(r"[0-9a-zA-Z]{10}",s) is None:
        print("Invalid")
        continue
    if len(re.findall(r"[A-Z]",s))<2:
        print("Invalid")
        continue
    if len(re.findall(r"\d",s))<3:
        print("Invalid")
        continue
    print("Valid")


# Regex and Parsing - Regex Substitution
import re

def modifier(match):
    s= match.group(0)
    
    if "&&" in s:
        return "and"
    else:
        return "or"

for _ in range(int(input())):
    s = input()
    
    print(re.sub("(?<=\s)&{2}(?=\s)|(?<=\s)\|{2}(?=\s)",modifier,s))


# Regex and Parsing - Validating Postal Codes

# Numpy - Arrays
def arrays(arr):
    a = numpy.array(arr,float)
    return numpy.flip(a)

# Numpy - Zeros and Ones
import numpy

i = tuple(map(int,input().split()))
print(numpy.zeros(i, dtype = numpy.int))
print(numpy.ones(i, dtype = numpy.int))

# Numpy - Concatenate
import numpy


i = list(map(int,input().split()))
n,m,p = i[0],i[1],i[2]
nl = [ list(map(int,input().split()))for _ in range(n)]
ml = [ list(map(int,input().split()))for _ in range(m)]
arr1, arr2 = numpy.array(nl), numpy.array(ml)
print(numpy.concatenate((arr1,arr2)))

# Numpy - Transpose and Flatten
import numpy

n,l = int(input().split()[0]), []
for _ in range(n):
    l.append(list(map(int,input().split())))
    arr = numpy.array(l)
print(numpy.transpose(arr))
print(arr.flatten())

# Numpy - Shape and Reshape
import numpy

l = list(map(int,input().split()))
print(numpy.reshape(numpy.array(l),(3,3)))

# Numpy - Eye and Identity
import numpy

numpy.set_printoptions(legacy="1.13")

i = list(map(int,input().split()))
print(numpy.eye(*i))

# Numpy - Array Mathematics
import numpy

nm = list(map(int, input().split()))
a = [list(map(int, input().split())) for _ in range(nm[0])]
b = [list(map(int, input().split())) for _ in range(nm[0])]

a = numpy.array(a)
b = numpy.array(b)

print(numpy.add(a,b))
print(numpy.subtract(a,b))
print(numpy.multiply(a,b))
print(numpy.floor_divide(a,b))
print(numpy.mod(a,b))
print(numpy.power(a,b))

# Numpy - Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy="1.13")
v = numpy.array(list(map(float,input().split())))
print(numpy.floor(v))
print(numpy.ceil(v))
print(numpy.rint(v))

# Numpy - Sum and Prod
import numpy

nm = list(map(int,input().split()))
a = [list(map(int,input().split())) for _ in range(nm[0])]
v = numpy.array(a)

print(numpy.prod(numpy.sum(v,axis=0)))

# Numpy - Min and Max
import numpy

nm = list(map(int,input().split()))
a = [list(map(int,input().split())) for _ in range(nm[0])]
v = numpy.array(a)
print(numpy.max(numpy.min(v,axis=1)))

# Numpy - Mean, Var, and Std
import numpy

nm = list(map(int,input().split()))
v = numpy.array([list(map(int,input().split())) for _ in range(nm[0])])
print(numpy.mean(v,axis=1))
print(numpy.var(v,axis=0))
print(round(numpy.std(v),11))

# Numpy - Dot and Cross
import numpy


n = int(input())
a = numpy.array([list(map(int,input().split())) for _ in range(n)])
b = numpy.array([list(map(int,input().split())) for _ in range(n)])

print(numpy.dot(a,b))

# Numpy - Inner and Outer
import numpy

a = numpy.array(list(map(int,input().split())))
b = numpy.array(list(map(int,input().split())))
print(numpy.inner(a,b))
print(numpy.outer(a,b))

# Numpy - Polynomials
import numpy

p = list(map(float,input().split()))
x = int(input())
print(numpy.polyval(p,x))

# Numpy - Linear Algebra
import numpy

n = int(input())
m = [list(map(float,input().split())) for _ in range(n)]
print(round(numpy.linalg.det(m),2))

# ----- PROBLEM 2 - Algorithms ------------------------

# Birthday Cake Candles
def birthdayCakeCandles(candles):
    m = max(candles)
    return candles.count(m)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    # x1 + v1*t == x2 + v2*t
    # t = (x2-x1)//(v1-v2)
    try:
        if (x2-x1)%(v1-v2) == 0 and (x2-x1)/(v1-v2) >= 0:
            return "YES"
        else:
            return "NO"
    except Exception:   # Zero division
        return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising
def viralAdvertising(n):
    def sub_recursive_shares(n):
        if n == 1:
            return 5
        else:
            return sub_recursive_shares(n-1)//2*3
    cumulative = 0
    for s in range(1,n+1):
        cumulative += sub_recursive_shares(s)//2
    return cumulative
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Recursive Digit Sum
def superDigit(n, k):
    def inner_recursive_sum(s:str):
        if len(s) == 1:
            return s
        else:
            sum_ = sum(list(map(int,list(s))))
            return inner_recursive_sum(str(sum_))
    mid_res = inner_recursive_sum(n)    # 1-digit string
    res = ""
    for _ in range(k):
        res = inner_recursive_sum(mid_res+res) # sum of mid_res k times
    return int(res)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion Sort - Part 1
def insertionSort1(n, arr):
    v = arr[-1]
    for i in reversed(range(0,len(arr)-1)):
        if arr[i]>v:
            arr[i+1] = arr[i]
            print(" ".join(map(str,arr)))
            if i == 0 and arr[0]>v:
                arr[0] = v
                print(" ".join(map(str,arr)))
        else:
            arr[i+1] = v
            print(" ".join(map(str,arr)))
            break
    
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Insertion Sort - Part 2
def insertionSort2(n, arr):
    def sub_sort(arr,i):
        v = arr.pop(i)
        for t in range(len(arr)-1):
            if arr[t] < v and arr[t+1] > v:
                arr.insert(t+1,v)
                return arr
                break
            elif v < arr[0]:
                arr.insert(0,v)
                return arr
                break
            elif t == len(arr)-2:
                arr.append(v)        
                return arr
    
    for i in range(1,n):
        if arr[i] > arr[i-1]:
            print(" ".join(list(map(str,arr))))
        else:
            arr = sub_sort(arr,i)
            print(" ".join(list(map(str,arr))))
    ...

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)


