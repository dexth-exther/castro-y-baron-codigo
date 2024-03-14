# Uso de algoritmo de diagnóstico
En este repositorio se encuentra alojado el código y la documentación necesaria para la ejecución del Script diagnóstico para sensores Xovis y su calibración.

El Script llamado "algoritmo-diagnóstico.py" contiene una generación de token para el acceso a la plataforma Xovis.
Debe tener permiso para almacenar un archivo de texto llamado "token.txt"
Este token tiene una validez de 24 horas, por lo que, si se realiza la consulta en días diferentes, este archivo se actualizará.

Para iniciar, debe ingresar las direcciones MAC de cada sensor que necesita evaluar en la siguiente línea:

        # Lista de direcciones MAC, copielas tal cual aparecen en el API, 
        # agregue comillas y separe por comas.        
      
         macs = ["00:6E:02:00:2A:84", "00:6E:02:00:3A:A0", "00:6E:02:00:2F:E8" ]



El programa debe poder ejecutarse con o sin archivo de token previo. Este se sobreescribe

# Librerias necesarias
Tenga en cuenta que las librerías deben estar instaladas, para ello debe ejecutarse cada comando en el cmd de windows:
* Cv2
  
        pip install opencv-python
* Numpy

      pip install numpy

* Requests

      pip install requests

* Os
* matplotlib
  
      pip install matplotlib

* Scipy

      pip install scipy

* Skimage

      pip install scikit-image

* Pandas

      pip install pandas

# Salida

Verá que en la carpeta que haya seleccionado como _path_ empezarán a aparecer archivos pdf con el informe de cada sensor. Cada uno de estos archivos tendrá el nombre de las direcciones mac respectivas.
Si no encuentra el informe de algunos sensores, puede ser que estén apagados.
