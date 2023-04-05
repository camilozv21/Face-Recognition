# importing libraries
#Obtener información del proceso actual
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
#------------------------proces--------------------
import psutil

process = psutil.Process()

# initializing MTCNN and InceptionResnetV1 
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

#-----------------------Entrenamiento-------------------
# Put your training code here. <3
#-----------------------Entrenamiento-------------------

# Using webcam recognize face

# loading data.pt file
load_data = torch.load('Facenet_pythorch/data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture(0) 
count = 0
cpu_array = []
memory_array = []
elapsed_time_array = []
c = 1

while True:
    # Guardamos el tiempo actual antes de ejecutar el bloque de código
    start_time = time.time()
    

    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
                
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i] 
                
                original_frame = frame.copy() # storing copy of frame before drawing on it
                
                if min_dist<0.55:
                    frame = cv2.putText(frame, name+' '+str(min_dist), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)                
                    frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (0,255,0), 2)
                    # Guardamos el tiempo actual después de ejecutar el bloque de código
                    end_time = time.time()

                    # Calculamos la diferencia entre ambos tiempos para obtener el tiempo de ejecución
                    elapsed_time = end_time - start_time

                    # Imprimimos el tiempo de ejecución en segundos
                    # print("Tiempo de ejecución:", elapsed_time, "segundos")
                     #Obtener el consumo de CPU y memoria del proceso
                    cpu_usage = process.cpu_percent()
                    mem_usage = process.memory_info().rss

                    cpu_array.append(cpu_usage)
                    memory_array.append(mem_usage)
                    elapsed_time_array.append(elapsed_time)

                    # print("CPU usage:", cpu_usage/100)
                    # print("Memory usage:", (mem_usage/1024)/1024)

                    c += 1

                    promCPU = sum(cpu_array) / c
                    promMEM = sum(memory_array) / c
                    promTIME = sum(elapsed_time_array) / c
                   



                frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (0,255,0), 2)
                
                
    cv2.imshow("IMG", frame)
        
    
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break
        
    elif k%256==32: # space to save image
        print('Enter your name :')
        name = input()
        
        # create directory if not exists
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name)
            
        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))
        
print("El promedio del porcentaje de uso de la CPU para un total de {} frames o imagenes es de {}".format(c,promCPU))
print("El promedio del uso de la memoria para un total de {} frames o imagenes es de {}".format(c,promMEM))
print("El promedio del tiempo que se demora en analizar un frame o una imagen es de {}".format(promTIME))

cam.release()
cv2.destroyAllWindows()
    

"""dataset = datasets.ImageFolder('D:/UNIVERSIDAD/Diseño_mecatronico2/Faces_Recognition_project/image_data') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True) 
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0)) 
        embedding_list.append(emb.detach()) 
        name_list.append(idx_to_class[idx])        

# save data
data = [embedding_list, name_list] 
torch.save(data, 'dataaux.pt') # saving data.pt file
print("Modelo entrenado con exito")"""