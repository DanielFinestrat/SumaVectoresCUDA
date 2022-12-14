# SumaVectoresCUDA
Suma de vectores de cualquier tamaño con la GPU. Puedes sumar vectores rellenados aleatoriamente con tu GPU nVidia, la única limitacion son las propias características de la GPU.
El programa está preparado para funcionar con una GPU nVidia 960. Si tu GPU no es ésta, ejecuta el programa deviceQuery, includio en las Samples que vienen con CUDA, el cual te mostrará la información de tu GPU. Introduce el número máximo de Threads por Block en la variable maxThreadsPerBlock del programa y el numero maximo de Blocks por Grid en la variablemaxBlocksPerGrid. También puedes cambiar el tamaño de los vectores modificando la constante kNumElements (actualmente a 90.000.000).
Una vez ejecutes el programa, éste te dirá el tiempo que ha tardado en ejecutar las operaciones desde la CPU y la GPU, demostrando así que realizarlas en la GPU es inmensamente más rápido.

Este programa es parte de una práctica realizada en la asignatura Dispositivos e Infraestructuras para Sistemas Multimedia, del tercer curso de Ingeniería Multimedia en la Universidad de Alicante.

2016.
