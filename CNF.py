print("^ for and, + for or, ! for not, > for implies, = for biconditional")
exp=str(input("Enter the expression"))
l=list(exp)
def search(l):
    if '=' in l:
        return l.index('=')
    elif '>' in l:
        return l.index('>')
    elif '!' in l:
        return l.index('!')
    elif '^' in l:
        return l.index('^')
    elif '+' in l:
        return l.index('+')
    else:
        return None
def bi_rule(left,right):
    l=[]
    r=[]
    for i in left:
        l.append(i)
    l.append('>')
    for j in right:
        l.append(j)
    for i in right:
        r.append(i)
    r.append('>')
    for j in left:
        r.append(j)
    return l,r
def imp_rule(left,right):
    res=[]
    res.append('!(')
    for i in left:
        res.append(i)
    res.append(')+(')
    for j in right:
        res.append(j)
    res.append(')')
    return res
def distri(l):
    if l[1]=='^' and l[4]=='+':
        exp="("+l[0]+"^"+l[3]+")+("+l[0]+"^"+l[5]+")"
    elif l[1]=='+' and l[4]=='^':
        exp="("+l[0]+"+"+l[3]+")^("+l[0]+"+"+l[5]+")"
    else:
        exp=l
    return exp
pos=search(l)
left=l[:pos]
right=l[pos+1:]
if l[pos]=='=':
    print("Applying biconditional elimination")
    leftimp,rightimp=bi_rule(left,right)
    lInd=search(leftimp)
    rInd=search(rightimp)
    leftA=leftimp[:lInd]
    leftB=leftimp[lInd+1:]
    rightA=rightimp[:rInd]
    rightB=rightimp[rInd+1:]
    print("Applying implication elimination")
    leftExp=imp_rule(leftA,leftB)
    rightExp=imp_rule(rightA,rightB)
    fin="("+"".join(leftExp)+")^("+"".join(rightExp)+")"
    print(fin)
elif l[pos]=='>':
    print("Applying implication elimination")
    fin="".join(imp_rule(left,right))
    print(fin)
elif l[pos]=="^" or l[pos]=="+":
    print("Applying distributive rule")
    fin="".join(distri(l))
    print(fin)
