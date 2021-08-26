import cv2
import glob
import numpy as np
from numpy.lib.function_base import append
from pyzbar.pyzbar import decode

path = glob.glob("imagens\*.png") # obter todas as imagens presentes na pasta imagens
for img in path: # loop para acessar as imagens uma de cada vez
    imagem = cv2.imread(img) 
    copia_imagem = imagem.copy() # copia da imagem para guardar a imagem original
    copia_imagem2 = imagem.copy() # copia da imagem original que será utilizada para
    #Nessa parte, foi feito uma marcara para detectar a plataforma, pois são as cores que mais se distinguem, sendo possível separar a região do qrcode
    amarelo = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    amarelo_lo = np.array([25,0,0])
    amarelo_hi = np.array([60,255,255])#range das cores amarelas da plataforma
    mask = cv2.inRange(amarelo, amarelo_lo, amarelo_hi)
    roxo = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    roxo_lo = np.array([110,50,50])
    roxo_hi = np.array([130,255,255])#range das cores azuis/roxas da plataforma
    mask2 = cv2.inRange(roxo, roxo_lo, roxo_hi)
    #Foi feita uma mascara para cadas cores e depois juntada as partes em um xor para a plataforma inteira ficar branca
    result = cv2.bitwise_xor(mask, mask2)
    #Como algumas plataformas estavam muito proximas umas das outras, foi feita um karnel para diminuir a area branca para que seja possivel diferenciar uma da outra
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(result,kernel,iterations = 1)
    #A partir disso foi possível encontrar os contornos de cada plataforma separada
    contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    List = []
    #Com os contornos, é possível criar aproximaçoes de um poligono parar esses contornos, para que seja possivel definir melhor qual contorno deve ser utilizado ou descartado
    for contour in contours:
        perimetro = cv2.arcLength (contour, True)
        if perimetro > 700: #utilizado somente para pegar somente os contornos com perimetro grande (ignorar os pequenos que nao serao usados)
            approx =  cv2.approxPolyDP(contour, 0.002*perimetro, True)
            #cv2.drawContours(copia_imagem, [approx], -1, (0, 255, 0), 2) # não influencia no código, apenas mostra o desenho dos contornos
            List.append(approx)
    #com os contornos é possivel agora separar as imagens de cada plataforma
    for c in List:
        x,y,w,h = cv2.boundingRect(c) #montando um retangulo em volta do contorno
        cropped = copia_imagem[y:y+h, x:x+w] #cortando a imagem 
        base = copia_imagem2[y:y+h, x:x+w] #copia do corta de imagem
        # fazendo um mascara agora para diferenciar o qrcode da plataforma
        black3 = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        black_lo3 = np.array([0,0,0])
        black_hi3 = np.array([0,0,255]) #range das cores branco ate preto
        mask3 = cv2.inRange(black3, black_lo3, black_hi3)
        #entao é possivel encontrar os contornos somente do qrcode
        contours2, _ = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        List2 = []
        #fazendo um novo loop para aproximar esses contornor de um poligono
        for contour2 in contours2:
            perimetro2 = cv2.arcLength (contour2, True)
            if perimetro2 > 100: #para pegar contornos com o tamanho esperado
                approx2 =  cv2.approxPolyDP(contour2, 0.002*perimetro2, True)
                #cv2.drawContours(base, [approx2], -1, (0, 255, 0), 2)
                List2.append(approx2)
        #sendo assim agora é possivel cortar o mais proximo possivel do qrcode
        for a in List2:
            x,y,w,h = cv2.boundingRect(a)
            cropped2 = base[y:y+h, x:x+w] #entao a imagem é cortada para separar somento o qrcode
            cropped2 = cv2.resize(cropped2, (cropped2.shape[1]*4,cropped2.shape[0]*4), interpolation=cv2.INTER_CUBIC) # dando resize para facilitar a visualizaçao do qrcode
            # fazendo uma ultima mascara para transformar todas as cores que não fazem parte do qrcode em preto
            black4 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2HSV)
            black_lo4 = np.array([0,0,70])
            black_hi4 = np.array([0,0,255])
            mask4 = cv2.inRange(black4, black_lo4, black_hi4)
            # realizando uma binarizaçao da imagem para que seja possivel traçar as bordas e assim conseguir os contornos das partes brancas do qrcode
            _, thresh = cv2.threshold(mask4, 0,255, cv2.THRESH_BINARY)
            edged = cv2.Canny(thresh, 20, 250)
            contours3, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            # com os contornos de cada qrcode foi feito um contorno envolvendo todos os outros, sendo assim possivel aproximar somente do quadrado o qrcode
            if len(contours3) > 3: # condiçao, pois algumas imagens existiam bordas sem qrcode na imagem
                for c in contours3:
                    x,y,w,h = cv2.boundingRect(c)
                    h = h+1 # para conseguir ler os q tavam bugados, alguns precisavam de um pedaço de borda ainda para baixo
                    boxes.append([x,y, x+w, y+h])
                # para isso foi pego os pontos mais abaixo, mais acima, mais a esquerda e mais a direita dessas bordas e assim gerando uma borda que envolva todasz elas
                boxes = np.asarray(boxes)
                left, top = np.min(boxes, axis=0)[:2]
                right, bottom = np.max(boxes, axis=0)[2:]
                #cv2.rectangle(cropped2, (left, top), (right, bottom), (36,255,12), 2)
                cropped4 = cropped2[top:bottom, left:right] # por fim cortando somente o quadrado central do qrcode
                # com o quadrado cortado, é possivel ler a informaçao presente em cada um deles
                qrcode = decode(cropped4)
                for i in qrcode:
                    print(i.data.decode("utf-8")) # printa a informaçao de cada qrcode
                cv2.imshow("fotos", cropped4) # por fim mostra a imagem final de cada qrcode
                cv2.waitKey(0)
                cv2.destroyAllWindows()