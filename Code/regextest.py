# Greedy vs. non-greedy
import re
x = '19'
#y = re.findall('^F.+?:', x) # non-greedy
y = re.match('[1-9]{1}', x)  # Starting with F, all characters (including whitespace),
if y == None:
    print "False"
else:
    print "True"
print y                     # one ore more characters, to the last colon --> greedy
