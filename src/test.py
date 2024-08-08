from molar_python import *

f = FileHandler.open('../molar/tests/protein.pdb')
topst = f.read()
src = Source(*topst)


sel1 = src.select_all()
sel2 = src.select_str("resid 5:6")

print("com1:",sel1.com())
print("com2:",sel2.com())


print("py1:",sel2.nth_pos(0))
sel2.nth_pos(0)[0] = 42.0
print("py2:",sel2.nth_pos(0))

#for el in sel2:
#    el.pos[0] = 42.0

#for el in sel2:
#    print(el.index, el.name, el.pos)
