import pandas as pd
import csv

speed_div = 20

angle_y_div = 180
angle_z_div = 1
local_x_div = -9
local_y_div = -1
local_z_div = -65

def angle_fix(x):
 if x< 180:
  return x
 else:
  return x-360


for i in range(1000):
    datas = []
    with open("./dataset_new/trial"+str(i)+".csv",newline='') as trials:
        data = csv.reader(trials)
        for j in data:
            datas.append(j)
            #print(i)


    with open("./post_dataset2/trial"+str(i)+".csv", 'w', newline='') as post_trials:
        writer = csv.writer(post_trials)
        writer.writerow(
        ["NPC_speed", "NPC_Angle_X","NPC_Angle_Y","NPC_Angle_Z","NPC_location_X", "NPC_location_Y", "NPC_location_Z",
         "AV_speed", "AV_angle_X", "AV_angle_Y", "AV_angle_Z","AV_location_X", "AV_location_Y", "AV_location_Z", "NPC_target_speed","NPC_turning_cmd",

         "predict_NPC_speed","predict_NPC_Angle_X","predict_NPC_Angle_Y","predict_NPC_Angle_Z","predict_NPC_location_X","predict_NPC_location_Y","predict_NPC_location_Z",
         "predict_AV_speed","predict_AV_Angle_X","predict_AV_Angle_Y","predict_AV_Angle_Z","predict_AV_location_X","predict_AV_location_Y","predict_AV_location_Z"]
        )
        #for i in range(4):
        #    writer.writerow([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        x=1
        while x < 32:
            k= datas[x]

            #print(k)
            npc_speed = float(k[0])/ speed_div
            npc_angle = k[1][7:-1].split(",")
            npc_location = k[2][7:-1].split(",")
            #print(npc_angle)
            npc_angle_x = angle_fix(float(npc_angle[0]))
            npc_angle_y = angle_fix(float(npc_angle[1]))
            npc_angle_z = angle_fix(float(npc_angle[2]))
            npc_location_x = float(npc_location[0])/local_x_div
            npc_location_y = float(npc_location[1])/local_y_div
            npc_location_z = float(npc_location[2])/local_z_div

            av_angle = k[4][7:-1].split(",")
            av_location = k[5][7:-1].split(",")

            av_angle_x = angle_fix(float(av_angle[0]))
            av_angle_y = angle_fix(float(av_angle[1]))
            av_angle_z = angle_fix(float(av_angle[2]))
            av_location_x = float(av_location[0])/local_x_div
            av_location_y = float(av_location[1])/local_y_div
            av_location_z = float(av_location[2])/local_z_div
            av_speed = float(k[3]) / speed_div
            #print(k[6])
            npc_cmd = k[6][1:-1].split(",")
            npc_target_speed =float(npc_cmd[0])/speed_div
            npc_turning_cmd = float(npc_cmd[1])

            p = datas[x+1]
            predicted_npc_speed = float(p[0])/speed_div
            predicted_npc_angle = p[1][7:-1].split(",")
            predicted_npc_location = p[2][7:-1].split(",")

            predicted_npc_angle_x = angle_fix(float(predicted_npc_angle[0]))
            predicted_npc_angle_y = angle_fix(float(predicted_npc_angle[1]))
            predicted_npc_angle_z = angle_fix(float(predicted_npc_angle[2]))
            predicted_npc_location_x = float(predicted_npc_location[0])/local_x_div
            predicted_npc_location_y = float(predicted_npc_location[1])/local_y_div
            predicted_npc_location_z = float(predicted_npc_location[2])/local_z_div

            predict_av_speed = float(p[3])/speed_div
            predicted_AV_angle = p[4][7:-1].split(",")
            predicted_AV_location = p[5][7:-1].split(",")
            predicted_AV_angle_x = angle_fix(float(predicted_AV_angle[0]))
            predicted_AV_angle_y = angle_fix(float(predicted_AV_angle[1]))
            predicted_AV_angle_z = angle_fix(float(predicted_AV_angle[2]))
            predicted_AV_location_x = float(predicted_AV_location[0])/local_x_div
            predicted_AV_location_y = float(predicted_AV_location[1])/local_y_div
            predicted_AV_location_z = float(predicted_AV_location[2])/local_z_div






            writer.writerow([npc_speed,npc_angle_x,npc_angle_y,npc_angle_z,npc_location_x,npc_location_y,npc_location_z,
                            av_speed,av_angle_x,av_angle_y,av_angle_z,av_location_x,av_location_y,av_location_z,npc_target_speed,npc_turning_cmd,

                            predicted_npc_speed,predicted_npc_angle_x,predicted_npc_angle_y,predicted_npc_angle_z,
                            predicted_npc_location_x,predicted_npc_location_y,predicted_npc_location_z,
                            predict_av_speed,predicted_AV_angle_x,predicted_AV_angle_y,predicted_AV_angle_z,
                            predicted_AV_location_x,predicted_AV_location_y,predicted_AV_location_z
                             ]
                            )
            x+=1






