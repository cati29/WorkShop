import cv2
import time
import numpy as np

Colors = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)] 

class_names = []

with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#capturando video
cap = cv2.VideoCapture("animal.mp4")

#chamar o opencv
net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

#criar modelo de detecção a partir do parâmetros carregados da rede
model = cv2.dnn_DetectionModel(net)

#carrega padrao de entrada(resolucao e escala) utilizados no modelo
model.setInputParams(size=(416,416),scale=1/255)

while True:
    x,frame = cap.read()

    start = time.time()

    classes,scores,boxes = model.detect(frame,0.1,0.3)

    end = time.time()
#color pega o color la de cima e efetua uma operacao matematica, onde a mesma classe id tera a mesma cor
#label adiciona a classe id detectada ao nome(pegando pelo id do arquivo de nome).
#cv2.retangle(desenhando a box no frame,a cor e a espessura)#cv2.puttext - escrevendo no frame(-10 acima da box)
    for (classid,score, box) in zip(classes,scores,boxes):
        color = Colors[int(classid) % len(Colors)]
        label = f"{class_names[classid]}:{score}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0],box[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #efetuando a operação ate
    fps_label = f"FPS: {round((1.0/ (end-start)),2)}"
    cv2.putText(frame,fps_label,
    (0,25),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),3)

    cv2.imshow("detections",frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()


