import math
import sys
import pickle
import pandas as pd
import numpy as np
import os
import time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3
import numpy as np
import os
import util

class Car:
    def __init__(self, speed =0, x=0, y=0, z=0, angle_x=0, angle_y=0, angle_z=0):
        self.speed = speed
        self.location_x = x
        self.location_y = y
        self.location_z = z
        self.angel_x = angle_x
        self.angel_y = angle_y
        self.angel_z = angle_z
        self.speed_div = 20
        self.angle_y_div = 180
        self.angle_z_div = 1
        self.local_x_div = -9
        self.local_y_div = -1
        self.local_z_div = -65


    def update_speed(self,value):
        self.speed = value
        return

    def update_location_x(self,value):
        self.location_x = value
        return
    def update_location_y(self,value):
        self.location_y = value
        return
    def update_location_z(self,value):
        self.location_z = value
        return
    def update_angel_x(self,value):
        self.angel_x = value
        return
    def update_angel_y(self,value):
        self.angel_y = value
        return
    def update_angel_z(self,value):
        self.angel_z = value
        return



def findFitness(deltaDlist, dList):
       # The higher the fitness, the better.

       minD = min(dList)
       minDeltaD = min(deltaDlist)
       print(minDeltaD)
       print(minD)


       fitness = 0.5 * minD + 0.5 * minDeltaD

       return fitness * -1


def brakeDist(speed):
    dBrake = 0.0467 * pow(speed, 2.0) + 0.4116 * speed - 1.9913 + 0.5
    if dBrake < 0:
        dBrake = 0
    return dBrake

def findDistance(ego, npc):
	ego_x = ego.location_x * ego.local_x_div
	ego_y = ego.location_y * ego.local_y_div
	ego_z = ego.location_z * ego.local_z_div
	npc_x = npc.location_x * npc.local_x_div
	npc_y = npc.location_y * npc.local_y_div
	npc_z = npc.location_z * npc.local_z_div
	dis = math.pow(npc_x - ego_x , 2) + math.pow(npc_y - ego_y, 2) + math.pow(npc_z - ego_z, 2)
	dis = math.sqrt(dis)
	return dis



def DeltaD(ego,npc,maxint=350):


    d = findDistance(ego, npc) - 4.6 # 4.6 is the length of a car
    deltaD = maxint # The smaller delta D, the better
    deltaDFront = maxint
    deltaDSide = maxint
    ego_x = ego.location_x * ego.local_x_div
    ego_y = ego.location_y * ego.local_y_div
    ego_z = ego.location_z * ego.local_z_div
    npc_x = npc.location_x * npc.local_x_div
    npc_y = npc.location_y * npc.local_y_div
    npc_z = npc.location_z * npc.local_z_div


    # When npc is in front
    #if npc_x  + 3.5 > ego_x  and npc_x  + 15  ego_x :
    #if ego_z + 3.5 > npc_z  and  ego_x +15 < npc_z  and (abs(ego_x - npc_x)  < 2):
        #if npc_x  > ego.location_z  - 2 and npc.location_z  < ego.location_z  + 2:
            #deltaDFront = d - brakeDist(ego.speed * 20)

    if ego_z+ 4.6 < npc_z and ego_z + 15 > npc_z:
            if npc_x > ego_x - 2:
                deltaDFront = d - brakeDist(ego.speed * 20)

    """
    # When ego is changing line to npc's front
    if npc_x - 4.6 > ego_x and npc_x - 20 < ego_x:
        if npc_z + 2 > ego_z and npc_z - 2 < ego_z and (ego.angel_y < 269 or ego.angel_y > 271):
            deltaDSide = d - brakeDist(npc.speed * 20)
    """


    deltaD = min(deltaDSide, deltaDFront)
    #print(deltaD)
    return deltaD





def Single_Trial_1npc(model, seed, ego, npc, maxint , timestep = 5):
    for n_NPC in range(len(seed)):
        DeltaD_list = [maxint for i in range(timestep)]
        D_list = [maxint for i in range(timestep)]
        actions, _ = model.predict([npc.speed, npc.angel_x, npc.angel_y ,npc.angel_z, npc.location_x, npc.location_y,npc.location_z,
                                        ego.speed,ego.angel_x,ego.angel_y,ego.angel_z, ego.location_x,ego.location_y,ego.location_z, seed[n_NPC][0][0]/20,seed[n_NPC][0][1]])
        print(actions)

        mindeltad = maxint
        mind = maxint
        npc.update_speed(actions[0])
        npc.update_angel_x(actions[1])
        npc.update_angel_y(actions[2])
        npc.update_angel_z(actions[3])
        npc.update_location_x(actions[4])
        npc.update_location_y(actions[5])
        npc.update_location_z(actions[6])

        ego.update_speed(actions[7])
        ego.update_angel_x(actions[8])
        ego.update_angel_y(actions[9])
        ego.update_angel_z(actions[10])
        ego.update_location_x(actions[11])
        ego.update_location_y(actions[12])
        ego.update_location_z(actions[13])
        currentd = findDistance(ego,npc)
        currentdeltad = DeltaD(ego,npc)
        mindeltad = min(mindeltad,currentdeltad)
        mind = min(mind,currentd)


        for i in range(0 , timestep):

            if(i == 0):
                start_checker = 7
            else:
                start_checker = 8
                mindeltad = maxint
                mind = maxint


            for k in range(start_checker):
                print("start")
                print(actions)
                actions = np.append(actions,[seed[n_NPC][i][0]/20,seed[n_NPC][i][1]])
                actions, _ = model.predict(actions)
                print(actions)
                npc.update_speed(actions[0])
                npc.update_angel_x(actions[1])
                npc.update_angel_y(actions[2])
                npc.update_angel_z(actions[3])
                npc.update_location_x(actions[4])
                npc.update_location_y(actions[5])
                npc.update_location_z(actions[6])

                ego.update_speed(actions[7])
                ego.update_angel_x(actions[8])
                ego.update_angel_y(actions[9])
                ego.update_angel_z(actions[10])
                ego.update_location_x(actions[11])
                ego.update_location_y(actions[12])
                ego.update_location_z(actions[13])
                currentdeltad = DeltaD(ego,npc)
                currentd = findDistance(ego,npc)
                mindeltad = min(mindeltad,currentdeltad)
                mind = min(mind,currentd)

            DeltaD_list[i] = mindeltad
            D_list[i]  = mind

        fitness = findFitness(DeltaD_list, D_list)
    resultDic = {}
    print(fitness)
    resultDic['fitness'] = fitness + maxint
    resultDic['seed'] = seed
    resultDic['fault'] = None
    #print(resultDic)
    return resultDic





if __name__ == '__main__':
    """

    objPath = sys.argv[1]
    resPath = sys.argv[2]
    "loading seed"
    objF = open(objPath, 'rb')
    scenarioObj = pickle.load(objF)
    objF.close()
    """


    scenarioObj = [[[18.39914332369696, 1], [5.3149371853294465, 2], [1.898656506501952, 2], [15.264262622370676, 0], [4.082076538117048, 1]]]
    objPath = " "
    resPath =" "
    print("Simulating start.... Defining the scenario initial position, angle, speed: ")
    npc = Car(speed =0, x=-0.446078247494167, y=0, z=1.00323122464693, angle_x=0, angle_y=-0.0074768066410229, angle_z=0)
    ego = Car(speed =0, x=-0.826253361172146, y=0, z=0.84938483605018, angle_x=0, angle_y=-0.1484375, angle_z=0)
    model = TD3.load("../../codes/models/results.zip")
    resultDic = Single_Trial_1npc(model, scenarioObj, ego, npc, 350, timestep = 5)
    print("fitness in this scenario is ")
    print(resultDic)
    util.print_res(resultDic)

    print("finishing simulation")

    if os.path.isfile(resPath) == True:
        #print("true")
        os.system("rm " + resPath)
"""
f_f = open(resPath, 'wb')
pickle.dump(resultDic, f_f)
f_f.truncate()
f_f.close()
"""




