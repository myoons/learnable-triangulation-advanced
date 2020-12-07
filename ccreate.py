from mvn.utils.multiview import Camera
import numpy
import torch

def create_cam(num):
    if num==1:
        K=[[1124.83,0,1018.069],[0,1123.996,534.9457],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[-0.2572692,0.5643166,0.7844484],[-0.9308321,0.07331344,-0.3580177],[-0.259546,-0.8222968,0.5064227]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[1181.555,-223.6565,3214.04]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='54138969'
    elif num==2:
        K=[[1124.563,0,967.7123],[0,1123.253,512.639],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[0.1483106,-0.9873533,-0.05601296],[-0.9800903,-0.1391867,-0.1415988],[0.1320118,0.07589837,-0.9883383]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[-734.5952],[-345.3347],[4178.617]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='55011271'
    elif num==4:
        K=[[1124.592,0,935.112],[0,1121.993,533.8148],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[0.1610608,0.6487166,-0.7437918],[-0.8499792,-0.2918426,-0.4385924],[-0.5015923,0.7028475,0.5043912]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[1426.446],[-1043.317],[5750.805]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='58860488'
    elif num==6:
        K=[[1122.218,0,972.8661],[0,1120.647,528.6412],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[-0.02079385,-0.9501867,0.3109871],[-0.9657519,-0.06137666,-0.2521037],[0.2586329,-0.3055786,-0.9163684]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[-591.5411],[71.836],[3680.45]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='60457274'
    return Camera(R,t,K,None,name)

def create_batch(batchSize):
    retlist=[]
    names=[1,2,4,6]
    for i in names:
        applist=[]
        for j in range(batchSize):
            temp_camera = create_cam(i)
            temp_camera.update_after_resize((1080, 1920), (384, 384))
            applist.append(temp_camera)
        retlist.append(applist)
    return retlist

def finalize(batchSize):
    cameras=create_batch(batchSize)
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in cameras], dim=0).transpose(1, 0)
    proj_matricies_batch = proj_matricies_batch.float()
    return proj_matricies_batch
    

if __name__ == '__main__' :
    finalize()