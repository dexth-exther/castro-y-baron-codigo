import cv2
import numpy as np
import requests
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim
import pandas as pd

imagenes = []

###### desde aquí se obtiene el token
def obtener_token():
            #-----------------------------------------------------------------
    #Aquí se pide el token automaticamente.
    payload = "{\"client_id\":\"PuN9AkWI6TnSjH0552Az0YxNIjujT03U\",\"client_secret\":\"pe12aJOUakay9sbp20QL8fRCQnCla_JbZ46RiCPWV0s6kUZsX7yiLw50YHXLP1bH\",\"audience\":\"https://api.xovis.cloud/\",\"grant_type\":\"client_credentials\"}"
    headers = {'content-type': "application/json"}
    response = requests.post("https://login.xovis.cloud/oauth/token", payload, headers=headers)
    cadena=response.content.decode("utf-8")
    inicio = cadena.find(':"') + 2
    fin = cadena.find('",')
    token = cadena[inicio:fin]

    #guardar token en un archivo--------
    with open("token.txt", "w") as file:
        file.write(token)
        
    time.sleep(3)
    if os.path.isfile("token.txt"):
                print("archivo creado")
    else:
                print("no se ha podido crear el archivo")
    return token
 
def leer_token():
    with open("token.txt", "r") as file:
        token = file.read().strip()
    return token    

def verificar_token():
    #primero se verifica que el archivo de texto sí exista-----
    time.sleep(3)
    if os.path.isfile("token.txt"):
        # Leer el token desde el archivo
        with open("token.txt", "r") as file:
            token = file.read().strip()
        
        # Verificar si se leyó un token válido
        if token:
            print("El archivo existe y contiene el siguiente token:")
            print(token)
            headers = { 'Authorization': f"Bearer {token}" }
            # URL de la imagen
            #image_url = "https://vm49-centralus-device-control.xovis.cloud/api/tunnel/80:1F:12:D5:F4:AB/api/v5/singlesensor/images/stereo.png"
            # Realizar la solicitud GET para obtener la imagen       
            image_url = "https://api.xovis.cloud/devices/68:27:19:B6:4B:3F/tunnel/api/v5/singlesensor/images/stereo.png"
            
            response = requests.get(image_url, headers=headers)

            # Verificar si la solicitud fue exitosa (código de estado 200)
            if response.status_code != 401:
                print("el token es correcto")
                
                
            else:
                print("El archivo está vacío o no contiene un token válido.")
                obtener_token()
                leer_token()
                
            
    #si no existe, pide un token nuevo.          
    else:
        print("El archivo no existe. Se solicitó un nuevo token.")
        obtener_token()


### aquí termina el proceso de token
verificar_token()
token=leer_token()

def obtener_imagen(image_url, token):
    base_url = "https://api.xovis.cloud/devices"
    headers = {'Authorization': f"Bearer {token}"}
    url = f"{base_url}/{image_url}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        image_data = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    print("No se pudo obtener la imagen desde el enlace proporcionado:", response.status_code)
    return None



def Stereo_image(image, nombre_ventana):

    image = cv2.resize(image, (600, 400))

    alto, ancho = image.shape[:2]

    # Margen de recorte (1 centímetro en cada margen)
    margen = 100

    # Recortar la imagen
    imagen_recortada = image[margen:alto-50, margen:ancho-150]
    


    hsv_image = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([98, 205, 105])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)


    area_azul = cv2.bitwise_and(imagen_recortada, imagen_recortada, mask=mask_blue)
    gray_area_azul = cv2.cvtColor(area_azul, cv2.COLOR_BGR2GRAY)

    # Excluir los píxeles negros de la máscara azul
    gray_area_azul[gray_area_azul == 255] = 0
    std_dev_intensity_azul = np.std(gray_area_azul)
    
    umbral_homogeneidad_azul = 57


    # Calcular la cantidad total de píxeles
    total_pixeles = imagen_recortada.shape[0] * imagen_recortada.shape[1]

    # Calcular la cantidad de píxeles azules
    cantidad_azul = cv2.countNonZero(mask_blue)

    # Calcular el porcentaje de azul en la imagen
    porcentaje_azul = (cantidad_azul / total_pixeles) * 100

    imagenes.append((nombre_ventana, imagen_recortada, area_azul, gray_area_azul))

    return std_dev_intensity_azul, umbral_homogeneidad_azul, area_azul, imagen_recortada, gray_area_azul, porcentaje_azul



def Heigth_map(image, nombre_ventana):
    # Aquí va la lógica de procesamiento para la segunda imagen (Height Map)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color_range = np.array([0, 25, 25])
    upper_color_range = np.array([179, 255, 245])
    
    mask_color = cv2.inRange(image_hsv, lower_color_range, upper_color_range)
    result_image = cv2.bitwise_and(gray_image, gray_image, mask=mask_color)
    total_pixels_non_black = np.count_nonzero(result_image)

    mask = cv2.inRange(image_hsv, (23, 135, 126), (71, 235, 226))
        
    pixels_in_range = np.count_nonzero(mask)

    # Superponer las imágenes y calcular el porcentaje de píxeles iguales
    overlap = cv2.addWeighted(mask_color, 0.5, mask, 0.5, 0)


    porcentaje_en_rangos = (pixels_in_range / total_pixels_non_black) * 100
    umbral_porcentaje = 30

    return porcentaje_en_rangos, umbral_porcentaje, overlap



def Start_stop(image, nombre_ventana, token): 

    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convertir la imagen a HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de colores que deseas incluir (por ejemplo, cualquier color excepto gris)
    lower_color_range = np.array([0, 80, 80])
    upper_color_range = np.array([179, 255, 255])

    # Crear una máscara para las regiones de interés (excepto tonos de gris)
    mask_color = cv2.inRange(image_hsv, lower_color_range, upper_color_range)

    # Aplicar la máscara para aislar las regiones de interés
    result_image = cv2.bitwise_and(image, image, mask=mask_color)

    # Convertir la imagen resultante a escala de grises
    result_gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Encontrar las coordenadas de los puntos que no son grises
    non_zero_coordinates = np.argwhere(result_gray_image != 0)
    non_zero_y_coordinates = non_zero_coordinates[:, 0]

    # Calcular el histograma de las coordenadas verticales que no son cero
    hist, bins = np.histogram(non_zero_y_coordinates, bins=50, range=(0, image.shape[0]))

    # Restringir el histograma solo a las coordenadas verticales que no son cero
    hist_non_zero = hist[hist != 0]
    bins_non_zero = bins[:-1][hist != 0]

    # Definir la función bimodal
    def bimodal_distribution(x, mu1, sigma1, a1, mu2, sigma2, a2):
        return a1 * np.exp(-((x - mu1) / sigma1) ** 2) + a2 * np.exp(-((x - mu2) / sigma2) ** 2)

    # Encontrar los picos del histograma
    peaks, _ = find_peaks(hist_non_zero)

    # Obtener las ubicaciones de los picos
    peak_locations = bins_non_zero[peaks]

    # Estimar los valores iniciales de los parámetros basados en los picos
    num_peaks = len(peak_locations)
    if num_peaks >= 2:
        mu1_init = peak_locations[0]  # Ubicación del primer pico
        mu2_init = peak_locations[-1]  # Ubicación del último pico
        sigma_init = np.mean(np.diff(peak_locations))  # Diferencia media entre picos
        a_init = hist_non_zero[peaks[0]] * sigma_init * np.sqrt(2*np.pi)  # Estimación de la amplitud
        b_init = hist_non_zero[peaks[-1]] * sigma_init * np.sqrt(2*np.pi)  # Estimación de la amplitud
    else:
        # Si no hay suficientes picos, se utilizan valores por defecto
        mu1_init = 100
        mu2_init = 300
        sigma_init = 50
        a_init = 2000
        b_init = 2000

    # Ajustar la distribución bimodal a los datos, aumentando maxfev
    params, _ = curve_fit(bimodal_distribution, bins_non_zero, hist_non_zero, p0=[mu1_init, sigma_init, a_init, mu2_init, sigma_init, b_init], maxfev=100000)

    # Calcular el coeficiente de determinación (R cuadrado) para evaluar el ajuste
    residuals = hist_non_zero - bimodal_distribution(bins_non_zero, *params)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((hist_non_zero - np.mean(hist_non_zero)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)


    # Definir el rango de búsqueda en el centro del histograma
    center_start = int(len(hist_non_zero) * 0.25)  # Inicio del rango (25% del histograma)
    center_end = int(len(hist_non_zero) * 0.85)    # Fin del rango (75% del histograma)

    # Encontrar el valle más pronunciado en el centro del histograma
    valley_position = center_start + np.argmin(hist_non_zero[center_start:center_end])  # Posición del valle más pronunciado en el centro
    valley_value = hist_non_zero[valley_position]  # Valor del valle más pronunciado

    # Calcular la coordenada vertical correspondiente al valle más pronunciado
    valley_coordinate = bins_non_zero[valley_position]

    # Definir los umbrales de valle máximo según el rango de frecuencia
    # Obtener el índice del pico más alto
    highest_peak_index = np.argmax(hist_non_zero)
    # Obtener el valor del pico más alto
    highest_peak_value = bins_non_zero[highest_peak_index]
    highest_peak_frequency = hist_non_zero[highest_peak_index]
    

    if valley_value is not None and r_squared is not None:

     if highest_peak_frequency < 1500:
      threshold_valley = 400
     elif highest_peak_frequency <= 2500:
      threshold_valley = 700
     else:
      threshold_valley = 870
    
    return valley_value, r_squared, bins_non_zero, hist_non_zero, params, bimodal_distribution, result_image,valley_coordinate, threshold_valley,highest_peak_value





def Visual(image, nombre_ventana, image_url4, image5):
    # Obtener la imagen cargada
    image5 = imagen_pasada(image_url4)

    # Verificar si se cargó correctamente
    if image5 is not None:
        # Redimensionar las imágenes al mismo tamaño
        image = cv2.resize(image, (600, 400))
        image5 = cv2.resize(image5, (600, 400))

        # Convertir las imágenes a escala de grises (si es necesario)
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)

        # Calcular la diferencia estructural entre las dos imágenes
        diff_struct, _ = ssim(gray1, gray2, full=True)

        # Calcular el porcentaje de diferencia
        porcentaje_diferencia = (1 - diff_struct) * 100

        # Definir el umbral para determinar si las imágenes son similares
        umbral_similitud = 70  # Porcentaje de similitud mínimo para considerar las imágenes como similares

    return porcentaje_diferencia, umbral_similitud



def imagen_pasada(image_url):
    # Buscar la subcadena 'singlesensor/images/' en la URL
    start_index = image_url.find('singlesensor/images/')
    
    # Extraer la MAC del URL basada en la posición encontrada
    if start_index != -1:
        # Si se encuentra la subcadena, extraer la MAC después de ella
        start_index += len('singlesensor/images/')

        mac_address = image_url[start_index:start_index + 17].replace(':', '_') + '.jpg'
        
    else:
        # Si la subcadena no está presente, asumir que la MAC está al principio
        mac_address = image_url[:17].replace(':', '_') + '.jpg'
    
    carpeta = 'mac_images'  # Carpeta en el mismo directorio

    # Obtener lista de nombres de archivos en la carpeta
    archivos_en_carpeta = os.listdir(carpeta)

    if mac_address in archivos_en_carpeta:
        ruta_archivo = os.path.join(carpeta, mac_address)
        # Cargar la imagen
        image = cv2.imread(ruta_archivo)
        return image  # Devolver la imagen cargada
    else:
        print(f"El archivo {mac_address} no existe en la carpeta.")
        return None



def guardar_resultados_en_excel(resultados):
    df = pd.DataFrame(resultados, columns=['MAC', 'Mapa Stereo Image', 'Mapa de Alturas', 'Mapa Start Stop', 'Visor del sensor'])
    df.to_excel('resultados_imagenes.xlsx', index=False)
    return 'resultados_imagenes.xlsx'  # Devuelve el nombre del archivo Excel


def main():
    token = leer_token()
    if token:
        # Lista de direcciones MAC
        macs = [
         
         "00:6E:02:00:2A:84", "00:6E:02:00:3A:A0", "00:6E:02:00:2F:E8", "00:6E:02:00:3B:EC", "00:6E:02:00:2C:C0", "00:6E:02:00:2B:C4", "00:6E:02:00:2B:B4", "00:6E:02:00:2A:8C", "00:6E:02:00:2B:4C", "00:6E:02:00:2F:B8", "00:6E:02:00:2D:70", "00:6E:02:00:3B:F0", "00:6E:02:00:3B:2C", "00:6E:02:00:3A:C8", "00:6E:02:00:2A:58", "00:6E:02:00:3B:C4", "00:6E:02:00:3B:6C", "00:6E:02:00:3A:CC", "00:6E:02:00:3A:AC", "00:6E:02:00:3A:08", "00:6E:02:00:2A:AC", "00:6E:02:00:2D:78", "00:6E:02:00:30:88", "00:6E:02:00:32:08", "00:6E:02:00:32:74", "00:6E:02:00:33:74", "00:6E:02:00:33:88", "00:6E:02:00:34:38", "00:6E:02:00:34:74", "00:6E:02:00:30:04", "00:6E:02:00:34:64", "00:6E:02:00:35:30", "00:6E:02:00:35:34", "00:6E:02:00:35:C8", "00:6E:02:00:35:E0", "00:6E:02:00:35:E8", "00:6E:02:00:35:EC", "00:6E:02:00:37:58", "00:6E:02:00:37:9C", "00:6E:02:00:38:98", "00:6E:02:00:38:E8", "00:6E:02:00:39:50", "00:6E:02:00:39:70", "00:6E:02:00:39:78", "00:6E:02:00:39:80", "00:6E:02:00:39:84", "00:6E:02:00:2A:88", "00:6E:02:00:2F:84", "00:6E:02:00:32:7C", "00:6E:02:00:35:3C", "00:6E:02:00:37:54", "00:6E:02:00:39:7C", "00:6E:02:00:3A:80", "00:6E:02:00:3A:84", "00:6E:02:00:3B:20", "00:6E:02:00:3C:18", "00:6E:02:00:3C:1C", "00:6E:02:00:3C:20", "00:6E:02:00:3C:38", "00:6E:02:00:3C:48", "00:6E:02:00:3C:50", "00:6E:02:00:3C:5C", "00:6E:02:00:3D:08", "00:6E:02:00:3D:14", "00:6E:02:00:3D:18", "00:6E:02:00:3D:1C", "00:6E:02:00:3D:E4", "00:6E:02:00:3E:1C", "00:6E:02:00:3E:90", "00:6E:02:00:3E:C0", "00:6E:02:00:3E:C4", "00:6E:02:00:3F:38", "00:6E:02:00:3F:60", "00:6E:02:00:3F:78", "00:6E:02:00:3F:BC", "00:6E:02:00:3F:C0", "00:6E:02:00:3F:C4", "00:6E:02:00:3F:DC", "00:6E:02:00:40:24", "00:6E:02:00:40:2C", "00:6E:02:00:40:34", "00:6E:02:00:40:94", "00:6E:02:00:40:98", "00:6E:02:00:40:C0", "00:6E:02:00:40:DC", "00:6E:02:00:40:EC", "00:6E:02:00:40:F0", "00:6E:02:00:40:FC", "00:6E:02:00:41:60", "00:6E:02:00:41:64", "00:6E:02:00:41:68", "00:6E:02:00:41:94", "00:6E:02:00:41:9C", "00:6E:02:00:41:B8", "00:6E:02:00:42:50", "00:6E:02:00:42:8C", "00:6E:02:00:42:D8", "00:6E:02:00:43:08", "00:6E:02:00:43:64", "00:6E:02:00:43:68", "00:6E:02:00:43:88", "00:6E:02:00:43:90", "00:6E:02:00:43:98", "00:6E:02:00:43:9C", "00:6E:02:00:43:A0", "00:6E:02:00:43:A8", "00:6E:02:00:43:B0", "00:6E:02:00:43:E4", "00:6E:02:00:43:EC", "00:6E:02:00:43:F0", "00:6E:02:00:43:F4", "00:6E:02:00:43:F8", "00:6E:02:00:43:FC", "00:6E:02:00:44:30", "00:6E:02:00:44:34", "00:6E:02:00:44:38", "00:6E:02:00:44:3C", "00:6E:02:00:44:40", "00:6E:02:00:44:A4", "00:6E:02:00:45:40", "00:6E:02:00:45:70", "00:6E:02:00:46:10", "00:6E:02:00:46:18", "00:6E:02:00:46:28", "00:6E:02:00:46:50", "00:6E:02:00:46:60", "00:6E:02:00:46:B8", "00:6E:02:00:47:44", "00:6E:02:00:47:A8", "00:6E:02:00:4E:60", "00:6E:02:00:52:D8", "00:6E:02:00:53:2C", "00:6E:02:00:54:84", "00:6E:02:00:97:D4", "00:6E:02:00:98:24", "00:6E:02:00:9A:68", "00:6E:02:00:9A:7C"

          ]
        #random.shuffle(macs)

        resultados = []

        for mac in macs:
            mensaje5 = ""
            valley_value = None
            threshold_valley= None
            # Procesar la primera imagen (Stereo Image)
            image_url1 = f"{mac}/tunnel/api/v5/singlesensor/images/stereo.png"
            image1 = obtener_imagen(image_url1, token)
            if image1 is not None:
                std_dev_intensity_azul, umbral_homogeneidad_azul, area_azul, _, gray_area_azul, porcentaje_azul = Stereo_image(image1, "Stereo Image")

                if porcentaje_azul < 15:
                    mensaje = f"Stereo Image incorrecto, El área azul en la imagen muy baja. Porcentaje de azul: {porcentaje_azul:.2f}%"
                elif std_dev_intensity_azul < umbral_homogeneidad_azul:
                    mensaje = f"Stereo Image correcto. El área azul en la imagen es considerada homogénea, Desviación estándar: {std_dev_intensity_azul:.2f}"
                else:
                    mensaje = f"Stereo Image incorrecto. El área azul en la imagen no es considerada homogénea, Desviación estándar: {std_dev_intensity_azul:.2f}"
                

            # Procesar la segunda imagen (Height Map)
            image_url2 = f"{mac}/tunnel/api/v5/singlesensor/data/history/height_map.jpg"
            image2 = obtener_imagen(image_url2, token)
            if image2 is not None:
                porcentaje_en_rangos, umbral_porcentaje, overlap = Heigth_map(image2, "Height Map")

                if porcentaje_en_rangos > umbral_porcentaje:
                    mensaje2 = f"Height Map correcto. La mayoría de píxeles están en los rangos de color,  Porcentaje: {porcentaje_en_rangos:.2f}"
                else:
                    mensaje2= f"Height Map incorrecto. La mayoría de píxeles NO están en los rangos de color,  Porcentaje: {porcentaje_en_rangos:.2f}"


            # Procesar la tercera imagen (Start Stop)
            image_url3 = f"{mac}/tunnel/api/v5/singlesensor/data/history/start_stop.jpg"
            image3 = obtener_imagen(image_url3, token)
            if image3 is not None:
                valley_value, r_squared, bins_non_zero, hist_non_zero, params, bimodal_distribution, result_image, valley_coordinate,threshold_valley,highest_peak_value = Start_stop(image3, "Start Stop", token)

            
            if valley_value is not None and threshold_valley is not None and r_squared is not None:
             if r_squared < 0.50 or valley_value > threshold_valley:
                 mensaje3 = f"El mapa start stop es incorrecto ya que valle: {valley_value} mayor a {threshold_valley} o R cuadrado: {r_squared:.2f} < 0.50)."
             elif valley_value <= threshold_valley and r_squared >= 0.50:
                 mensaje3 = f"El mapa start stop es correcto (Valle: {valley_value} <= {threshold_valley} y R cuadrado: {r_squared:.2f} > 0.50). "
        
             if threshold_valley is None:
                    mensaje3 = "El valor de threshold_valley es None"


            # Procesar la cuarta imagen (Visual)
            image_url4 = f"{mac}/tunnel/api/v5/singlesensor/experimental/settings/scene/image.jpg"
            image5 = imagen_pasada(image_url4)  # Obtener la imagen cargada
            if image5 is not None:
                image4 = obtener_imagen(image_url4, token)
                if image4 is not None:
                    porcentaje_diferencia, umbral_similitud = Visual(image4, "Visual", image_url4, image5)

                    if porcentaje_diferencia == 0:
                     mensaje4 = f"Las imágenes son iguales. Porcentaje de diferencia: {porcentaje_diferencia:.2f}%"
                    elif porcentaje_diferencia > umbral_similitud:
                     mensaje4 = f"Las imágenes son diferentes. Porcentaje de diferencia: {porcentaje_diferencia:.2f}%"
                    else:
                     mensaje4 = f"Las imágenes son similares. Porcentaje de diferencia: {porcentaje_diferencia:.2f}%"


                    if (porcentaje_azul >= 15 and
                        std_dev_intensity_azul <= umbral_homogeneidad_azul and
                            porcentaje_en_rangos >= umbral_porcentaje and
                            valley_value <= threshold_valley and r_squared >= 0.50 and
                            porcentaje_diferencia <= umbral_similitud):
                        mensaje5=f"EL SENSOR '{mac}' NO DEBE SER CALIBRADO"

                    elif(porcentaje_azul < 20):
                        mensaje5=f"EL SENSOR '{mac}' DEBE SER CALIBRADO"
                        resultados.append((mac,mensaje,mensaje2,mensaje3,mensaje4))

                    else:
                        mensaje5=f"EL SENSOR '{mac}' DEBE SER CALIBRADO"
                        resultados.append((mac,mensaje,mensaje2,mensaje3,mensaje4))

                    # Agregar los resultados a la lista de resultados
                    
                    mensaje_combined = mensaje + '\n' + mensaje2 + '\n' + mensaje3+ '\n' +  mensaje4+ '\n' + mensaje5

                    # Mostrar las imágenes en la misma ventana
                    
                    plt.figure(figsize=(40, 30))
                    
                    # Stereo Image
                    plt.subplot(4, 3, 1)
                    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)) 
                    plt.title("Imagen Original, Stereo Image")

                    plt.subplot(4, 3, 2)
                    plt.imshow(cv2.cvtColor(area_azul, cv2.COLOR_BGR2RGB))
                    plt.title(f'Área Azul en Stereo Image')

                    plt.subplot(4, 3, 3)
                    plt.imshow(gray_area_azul, cmap='gray')
                    plt.title('Imagen en Escala de Grises')
                    

                    # Height Map
                    plt.subplot(4, 3, 4)
                    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)) 
                    plt.title("Imagen Original, Height Map")

                    plt.subplot(4, 3, 5)
                    plt.imshow(cv2.cvtColor(overlap, cv2.COLOR_BGR2RGB))
                    plt.title(f'Superposición')


                    # Start stop
                    plt.subplot(4, 3, 7)
                    plt.plot(bins_non_zero, hist_non_zero, color='blue', label='Puntos no Grises')
                    plt.plot(bins_non_zero, bimodal_distribution(bins_non_zero, *params), color='red', label='Distribución Bimodal Ajustada')
                    plt.xlabel('Coordenada Vertical')
                    plt.ylabel('Frecuencia')
                    plt.axvline(x=valley_coordinate, color='green', linestyle='--', label='Valle más pronunciado')
                    plt.text(valley_coordinate, max(hist_non_zero), f'valle: {valley_value}', color='black', fontsize=10, ha='right', va='top')
                    plt.legend()
                    plt.subplot(4, 3, 8)
                    plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
                    plt.title(f'Imagen Original, Start stop')
                    

                    plt.subplot(4, 3, 9)
                    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                    plt.title(f'Resultante')


                    # Visual
                    plt.subplot(4, 3, 10)
                    plt.imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
                    plt.title(f'visual presente')

                    plt.subplot(4, 3, 11)
                    plt.imshow(cv2.cvtColor(image5, cv2.COLOR_BGR2RGB))
                    plt.title(f'visual pasada')
                    plt.tight_layout()


                    plt.figtext(0.83, 0.1, mensaje_combined, wrap=True, horizontalalignment='center', fontsize=15, va='bottom')
                    mac_ = mac.replace(":", "_")
                    plt.savefig(f"{mac_}.pdf")
                    plt.close()
                    #plt.savefig('resultados.pdf')
                    #plt.show()

        # Guardar los resultados en un archivo Excel después de procesar todas las MAC
        archivo_excel = guardar_resultados_en_excel(resultados)
        if archivo_excel:
            print("El archivo Excel se ha guardado correctamente en la siguiente ubicación:")
            print(os.path.abspath(archivo_excel))
        else:
            print("Error al guardar el archivo Excel.")



if __name__ == "__main__":
    main()
