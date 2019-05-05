import math
import csv
import os

os.chdir(r"C:\Users\98270\Desktop\201903\06_ml\C6-Material")
k = 0
with open('shape_detect.csv', 'r') as handler:
    reader = csv.reader(handler) 
    for row in reader:
        if k>1:
            f = list(reader)
        k += 1     
print (f[0:5])

def geterror(f, alpha):
    pred = []
    obs_number = len(f)
    sum_e = 0
    for j in range(obs_number):
        P = float(f[j][0])
        L = float(f[j][1])
        net = (math.log(P)) - (math.log(L))*alpha
        act = math.exp(net)
        if act < 3.5:
            stype = 0
            pshape = 'T' 
        else:    
            stype = 1
            pshape = 'R' 
        if f[j][2] == 'T':
            e = stype
        if f[j][2] == 'R':    
            e = 1 - stype
        sum_e = sum_e + e    
        pred.append([f[j][2], pshape, e])   
    avg_e = sum_e / obs_number   
    x = [pred, avg_e]
    return x
  
alpha = 0.1
error = geterror(f, alpha)[1]
pred = geterror(f, alpha)[0]
print (error)
print (pred[0:10])


T = 200
for j in range(T):
    if error> 0.1:
        alpha = alpha + error*0.1    
        error = geterror(f, alpha)[1]
    else:   
        print ('iteration #',j)
        break

print ('final error ratio', error)        
print ('alpha:', alpha)      
prediction = geterror(f, alpha)[0]
print ('-------------prediation----------------------')
prediction
