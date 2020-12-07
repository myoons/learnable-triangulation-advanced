	from mvn.utils.multiview import Camera
import numpy
import torch

def create_cam(num):
    if num==1:
        K=[[1124.563,0,967.7123],[0,1123.253,512.639],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[0.1483106,-0.9873533,-0.05601296],[-0.9800903,-0.1391867,-0.1415988],[0.1320118,0.07589837,-0.9883383]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[-734.5952],[-345.3347],[4178.617]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='54138969'
    elif num==2:
        K=[[1118.958,0,971.2457],[0,1118.306,520.8431],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[0.3139955,-0.1641321,-0.9351296],[-0.9035717,-0.3540529,-0.2412564],[-0.2914875,0.9207101,-0.2594762]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[-709.3732],[-889.1614],[5026.707]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='55011271'
    elif num==4:
        K=[[1122.902,0,913.9825],[0,1122.022,518.5341],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[-0.1943118,0.9434211,0.2686998],[-0.9006306,-0.06303116,-0.4299902],[-0.3887254,-0.3255515,0.8619216]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[870.4645],[-213.2622],[3788.815]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='58860488'
    elif num==6:
        K=[[1124.197,0,930.3889],[0,1122.969,578.2481],[0,0,1]]
        K=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in K])
        R=[[-0.1066995,-0.7382388,0.6660471],[-0.9860531,-0.007487193,-0.1662626],[0.1277283,-0.674498,-0.7271438]]
        R=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in R])
        t=[[-38.31519],[-394.2763],[3723.456]]
        t=numpy.array([numpy.array([numpy.float32(i) for i in lst]) for lst in t])
        name='60457274'
    return Camera(R,t,K,None,name)

def create_batch(batchSize):
    retlist=[]
    names=[1,2,4,6]
    for i in names:
        applist=[]
        for j in range(batchSize):
            temp_camera = create_cam(i).update_after_resize((1080, 1920), (384, 384))
            applist.append(temp_camera)
        retlist.append(applist)
    return retlist

def finalize(batchSize):
    cameras=create_batch(batchSize)
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in cameras], dim=0).transpose(1, 0)
    device = torch.device(0)
    proj_matricies_batch = proj_matricies_batch.float().to(device)
    return proj_matricies_batch
    

if __name__ == '__main__' :
    finalize()