import math
def findinput(file_name="logs/Apollo5.0cluster2_3.log"):
    f = open(file_name,"r")
    lines = f.readlines()
    #print(lines[1])
    pre_list = []

    i = 0
    #print(len(lines))
    temp=[]
    while i < len(lines):
        temp.append(lines[i])
        if "=== Run simulation ===" in lines[i]:
            pre_list.append(temp)
            temp=[]
            #print(lines[i])
        i=i+1
    #print(pre_list[1])
    ego_list=[]
    npc_list=[]
    safe =[]
    for case in pre_list:
        for f in case:
            if "it is ego fault" in f:
                ego_list.append(case)
            elif "it is npc fault" in f:
                npc_list.append(case)
        if case not in ego_list or npc_list:
            safe.append(case)
    #print(ego_list)

    return [safe,ego_list,npc_list]

input = findinput()
#print(input[0])
#for i in input[1]:
#    print(i)
#print(input[2])

def angle_fix(x):
 if x< 180:
  return x
 else:
  return x-360


def extrat_data_input_npc(input,t):
    trueinput = input[t]
    #print(trueinput)
    true_label_data =[]
    speed =[]
    location =[]
    angle = []
    label =[]
    i =0
    while i<len(trueinput):
        #print(trueinput[i])
        j = 0
        speeds=[]
        locations =[]
        angles =[]
        labels =[]
        while j< len(trueinput[i]):
            #label.append(t)

            if "= current npc speed" in trueinput[i][j] :
                #print((trueinput[i]))
                speeds.append(float(trueinput[i][j+1]))
            elif "= current npc location" in trueinput[i][j] :
                l= list((trueinput[i][j+1][7:][:-2].split(", ")))
                for x in range(0, len(l)):
                     l[x] = float(l[x])

                locations.append(l)

                #print(l)
            elif "= current npc angle" in trueinput[i][j] :
                l=(list(trueinput[i][j+1][7:][:-2].split(", ")))
                for x in range(0, len(l)):
                     l[x] =angle_fix(float(l[x]))
                angles.append(l)
            elif "= iscollide" in trueinput[i][j] :
                #print((trueinput[i]))
                labels.append(int(trueinput[i][j+1]))

            j+=1
        speed.append(speeds)
        location.append(locations)
        angle.append(angles)
        label.append(labels)
        #print(len(speeds))
        i+=1
    #print(speed)
    return [speed,location,angle,label]
def extrat_data_input_ego(input,t):
    trueinput = input[t]
    #print(trueinput)
    true_label_data =[]
    speed =[]
    location =[]
    angle = []
    label =[]
    i =0
    while i<len(trueinput):
        #print(trueinput[i])

        j = 0
        speeds=[]
        locations =[]
        angles =[]
        labels =[]
        while j< len(trueinput[i]):

            if "= current ego speed" in trueinput[i][j] :
                speeds.append(float(trueinput[i][j+1]))
            elif "= current ego location" in trueinput[i][j] :
                l= list((trueinput[i][j+1][7:][:-2].split(", ")))
                for x in range(0, len(l)):
                     l[x] =float(l[x])
                locations.append(l)

            elif "= current ego angle" in trueinput[i][j] :
                l=(list(trueinput[i][j+1][7:][:-2].split(", ")))
                for x in range(0, len(l)):
                     l[x] =angle_fix(float(l[x]))
                angles.append(l)
            elif "= iscollide" in trueinput[i][j]:
                #print((trueinput[i]))
                labels.append(int(trueinput[i][j+1]))
            j+=1
        speed.append(speeds)
        location.append(locations)
        angle.append(angles)
        label.append(labels)
        #print(len(speeds))
        i+=1
    #print((speed))
    return [speed,location,angle,label]

t= 1
npc_true = extrat_data_input_npc(input, t)
#print(a[0])
print(len(npc_true[1]))
ego_true = extrat_data_input_ego(input, t)
print(len(ego_true[1]))

#print(ego_true[2])
#example of feature
# total number of true input 201
# total number of false input
t= 0
npc_false = extrat_data_input_npc(input, t)
print(len(npc_false[1]))
ego_false = extrat_data_input_ego(input, t)
print(len(ego_false[1]))
#print(npc_false)

#print( (ego_false[1][-6]))

"""
generate data D, A, and direction 
"""
def generate_D(npc_location,ego_location):
    D = abs(ego_location[0] - npc_location[0]) +  abs(ego_location[1] - npc_location[1]) +  abs(ego_location[2] - npc_location[2])
    return D

def generate_A(npc_angle,ego_angle):
    A = abs(ego_angle[0] - npc_angle[0]) +  abs(ego_angle[1] - npc_angle[1]) + abs(ego_angle[2] - npc_angle[2])
    return A

def generate_direction(npc_location,ego_location,ego_angle):
    dot = ego_location[0] * npc_location[0] + ego_location[2] + npc_location[2]
    det = ego_location[0] * npc_location[0] - ego_location[2] + npc_location[2]
    angle = math.atan2(dot, det)
    if ego_location[2] > npc_location[2]:
        if ego_angle > 180:
            return angle + (360 - math.radians(ego_angle))
        else:
            return angle -  math.radians(ego_angle)
    else:
        if ego_angle> 180:
            return 3.1415926 - angle + (360 - math.radians(ego_angle))
        else:
            return 3.1415926 - (angle -  math.radians(ego_angle))

def add_last(l):
    #i =len(l[0][0])
    #print(l[0][0])
    #print(i)
    #print(len(l[1][1]))
    #print(l[0])
    #print(len(l[0]))
    for x in l[0]:
        i = len(x)
        if (x ==[]):

            pass

        #print(x[1])
        else:
            while i<40:
                #print(x)
                x.append(x[-1])
                i+=1
    for x in l[1]:
        i = len(x)
        #print(x[1])
        if (x ==[]):
            pass

        #print(x[1])
        else:

            while i<40:
                x.append(x[-1])
                i+=1

    for x in l[2]:
        i = len(x)
        if (x ==[]):
            pass


        #print(x[1])
        else:
            while i<40:
                x.append(x[-1])
                i+=1
    for x in l[3]:
        i = len(x)
        if (x ==[]):
            pass


        #print(x[1])
        else:
            while i<40:
                x.append(x[-1])
                i+=1


    return l

def pop0(l):
    #print(type(l))
    l[0]=l[0][1:]
    l[1]=l[1][1:]
    l[2]=l[2][1:]
    return l

#print(npc_false)


#print
import csv
def csvdata(npc,ego,i,j):
    #print(len(npc[2][i]))
    location_mhd = generate_D([(npc[1][i][j][0]),(npc[1][i][j][1]),(npc[1][i][j][2])],[(ego[1][i][j][0]),(ego[1][i][j][1]),(ego[1][i][j][2])])
    angle_mhd = generate_D([(npc[2][i][j][0]),(npc[2][i][j][1]),(npc[2][i][j][2])],[(ego[2][i][j][0]),(ego[2][i][j][1]),(ego[2][i][j][2])])
    clock =  generate_direction([(npc[1][i][j][0]),(npc[1][i][j][1]),(npc[1][i][j][2])] , [(ego[1][i][j][0]),(ego[1][i][j][1]),(ego[1][i][j][2])] , ego[2][i][j][0])
    return [npc[0][i][j],
            ego[0][i][j],
            location_mhd,
            angle_mhd,
            clock,  npc[3][i][j]]

header = ['npc_speed',
         'ego_speed',
         'location_MHD',
         'angel_MHD',
         'clock',
         'label']

#a  = csvdata(npc_true,ego_true,0,1)

#print(a)
i=  0
while i< 400:
    with open(('5.0cluster_2_2/input' + str(i)+ '.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        j = 0
        print(i)

        while j < len(npc_true[0][i]):

            data = csvdata(npc_true,ego_true,i,j)
            #print(data)
            #if (data[-1] == 1):
            writer.writerow(data)
                #break
            #else:
             #   writer.writerow(data)
            #print(type(data[2]))
            j=j+1
    i=i+1




y = 400

print(y)

i=  0
while i< 200:
    with open(('5.0cluster_2_2/input' + str(i+y)+ '.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        j = 0
        #print(len(npc_false[0]))
        while j < len(npc_false[0][i]):

            data = csvdata(npc_false,ego_false,i,j)
            #print(data)
            #if (data[-1] == 1):
            writer.writerow(data)
                #break
            #else:
             #   writer.writerow(data)
            #print(type(data[2]))
            j=j+1
    i=i+1
'''
#z = y+i

while i< 200:
    with open(('3.5cluster_cluster2/input' + str(i+y)+ '.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        j = 0
        print(len(npc_false[0]))
        while j < len(npc_false[0][i]):

            data = csvdata(npc_false,ego_false,i,j)
            #print(data)
            #if (data[-1] == 1):
            writer.writerow(data)
                #break
            #else:
             #   writer.writerow(data)
            #print(type(data[2]))
            j=j+1
    i=i+1

'''
