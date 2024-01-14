import numpy
a=numpy.loadtxt(r"C:\Users\\GBDJ\Desktop\index.txt",encoding="UTF-8",dtype=str,delimiter="\t")
x={i:0 for i in range(5,0,-1)}
for j in range(1,len(a)):
    temp=a[j]
    for i in range(len(temp)):
        if temp[i]=='√':
            x[int(a[0][i])]+=1
    print("\t".join([str(x[z]) for z in range(5,0,-1)]))
    x={i:0 for i in range(5,0,-1)}
# for j in range(5):
#     for i in range(len(a)):
#         print((a[i][j:85:5]=="√").sum())
#     print("+")