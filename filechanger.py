import os
import shutil
import re

# s01~ subjects
# act01~act?? actions
# subact action number


"""
action 1->
action 2-> Directions
action 3-> Discussion
action 4-> Eating
action 5-> Greeting
action 6-> Phoning
action 7-> Posing
action 8-> Purchases
action 9-> Sitting
action 10-> SittingDown
action 11-> Smoking
action 12-> Photo
action 13-> Waiting
action 14-> Walking
action 15-> WalkDog
action 16-> WalkTogether
"""
temp=['','','Directions','Discussion','Eating','Greeting','Phoning','Posing','Purchases','Sitting','SittingDown','Smoking','TakingPhoto','Waiting','Walking','WalkingDog','WalkingTogether']
temp2=['54138969', '55011271', '58860488', '60457274']
cwd=os.getcwd()
images=os.listdir('images')
for f1 in images:
    for f2 in os.listdir('images/'+f1):
        if f2.split('.')[1]=='jpg':
            f3= f2.split('_'); # f3[1] determines subject, f3[3] determines action, f3[5] determines subaction, f4[7] determines camera, f4[8] determines imagenum
            subname='S'+str(int(f3[1]))
            actname=temp[int(f3[3])]
            subact='-'+str(int(f3[5]))
            camnum=temp2[int(f3[7])-1]
            if len(f3)==10:
                continue
            imgnum=f3[8]
            if not os.path.exists(cwd+'/data/human36m/processed/'+subname+'/'+actname+subact+'/imageSequence/'+camnum):
                os.makedirs(cwd+'/data/human36m/processed/'+subname+'/'+actname+subact+'/imageSequence/'+camnum)
            os.replace(cwd+'/images/'+f1+'/'+f2,cwd+'/data/human36m/processed/'+subname+'/'+actname+subact+'/imageSequence/'+camnum+'/'+'img_'+imgnum)
                