/*
	Daniel Finestrat Martinez, 48719584H.
	Vamos a pasar de hacer 1 operacion con CPU a 25600 o más al mismo tiempo con GPU
	
	Para cambiar el tamanyo del vector cambiar kNumElements
	Para cambiar el numero maximo de hilos por bloque usar maxThreadsPerBlock
	Para cambiar el numero maximo de bloques por grid usar maxBlocksPerGrid

	CUDA Device Query en "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\1_Utilities\deviceQuery"
	960 Max Threads per Block = 1024
	960 Max Blocks per Grid = 65536 (Usamos 65535)

	CPU = HOST
	GPU = DEVICE
	Kernel = isntrucciones de GPU que se ejecuta en un millon o dos millones de copias diferentes a la vez
	__global__ = Se ejecuta en GPU pero se llama desde CPU
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <Windows.h>

// Retorna (a - b) en segundos
double performancecounter_diff(LARGE_INTEGER *a, LARGE_INTEGER *b) {
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return (double)(a->QuadPart - b->QuadPart) / (double)freq.QuadPart;
}

//Suma de vectores de mucho tamanyo, sea el numero elementos divisible entre 2 o no
__global__ void kernel_suma_vectores(const float * cpA, const float * cpB, float * cpC, const int kNumElem) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int salto = blockDim.x * gridDim.x; //Metemos un salto para que cada hilo ejecute mas de una operacion 
	for ( ; i < kNumElem; i += salto) { cpC[i] = cpA[i] + cpB[i]; }
}

int main(){

	LARGE_INTEGER t_ini, t_fin;	//Tiempo inicial y final de Proceso
	double secsH; //Tiempo de operacion

	//PASO 1: INICIALIZACION
	cudaSetDevice(0); //Llamamos a cualquier variable del kernel para inicializar, esta nos dice q usemos la GPU 0
	
	//PASO 2: DECLARACION
	//Apuntamos el tamanyo maximo de bloque y de grid de nuestra grafica
	std::cerr << "Creamos los Vectores.\n\n";

	const int maxThreadsPerBlock = 1024; //Ajustar esta cifra segun tu GPU
	const int maxBlocksPerGrid = 65535; //Ajustar esta cifra segun tu GPU



	const int kNumElements = 90000000; //Apuntamos numero de elementos de cada vector que sumaremos (ej. 25600)
	size_t vector_size_bytes = kNumElements * sizeof(float); //Obtenemos cantidad de memoria necesaria para un vector del numero de elementos definido, siendo variables float

	//Reservamos vector_size_bytes de memoria para cada vector que vayamos a usar em la CPU (Host)
	float *h_A_ = (float *)malloc(vector_size_bytes);
	float *h_B_ = (float *)malloc(vector_size_bytes);
	float *h_C_ = (float *)malloc(vector_size_bytes);
	if (h_A_ == NULL || h_B_ == NULL || h_C_ == NULL) { std::cerr << ("MEMORIA MAL RESERVADA"); getchar(); exit(-1); } //Comprobamos errores

	//Ahora declaramos los mismos pero en la grafica y reservamos la memoria
	float *d_A_ = NULL; float *d_B_ = NULL; float *d_C_ = NULL;
	cudaMalloc((void **)&d_A_, vector_size_bytes); //Puntero a puntero para reservar memoria?
	cudaMalloc((void **)&d_B_, vector_size_bytes);
	cudaMalloc((void **)&d_C_, vector_size_bytes);

	std::cerr << "Rellenamos los Vectores.\n\n";
	//Inicializamos los vectores a numeros aleatorios
	for (int i = 0; i < kNumElements; ++i) { //++i mas eficiente que i++
		h_A_[i] = rand() / (float)RAND_MAX;
		h_B_[i] = rand() / (float)RAND_MAX;
	}

	//PASO 3: TRANSFERENCIA
	std::cerr << "Transferimos los datos al Device.\n\n";
	cudaMemcpy(d_A_, h_A_, vector_size_bytes, cudaMemcpyHostToDevice); //Copia de CPU a GPU
	cudaMemcpy(d_B_, h_B_, vector_size_bytes, cudaMemcpyHostToDevice);

	//PASO 4: EJECUCION
	std::cerr << "Empezamos la Ejecucion\n\n";
	int threadsPerBlock =  maxThreadsPerBlock; //Ponemos el tamanyo al maximo para minimizar los bloques necesarios

	/*
	Redondeamos el numero de bloques por grid HACIA ARRIBA para evitar desbordamiento con vectores arbitrarios.
	Para ello usamos la formula z = 1 + ((x - 1) / y) que automaticamente nos sube el numero INTEGRER al siguiente
	cuando nos pasamos. De este modo:
		1 + ((25600 - 1) / 256) = 100.99 = 100 (ya que INT trunca)
		1 + ((25601 - 1) / 256) = 101
	*/
	int blocksPerGrid = 1 + ((kNumElements - 1) / threadsPerBlock);
	if (blocksPerGrid > maxBlocksPerGrid) { blocksPerGrid = maxBlocksPerGrid; } //Si blocksPerGrid es mayor que el que podemos usar, lo dividimos para obtener el maximo numero de bloques por grid admitido

	dim3 block(threadsPerBlock, 1, 1); //Definimos los bloques, x sera los hilos que contiene cada bloque
	dim3 grid(blocksPerGrid, 1, 1); //Definimos la grid con 3 variables, x sera los bloques que contiene cada grid
	
	kernel_suma_vectores<<<grid, block>>> (d_A_, d_B_, d_C_, kNumElements); //Invocacion de kernel. Le pasamos variables de DISPOSITIVO (GPU)

	//Comprobamos si tenemos errores
	cudaError_t err_ = cudaGetLastError();
	if (err_ != cudaSuccess) {
		std::cerr << cudaGetErrorString(err_);
		getchar(); exit(-1);
	}

	//PASO 5: TRANSFERENCIA
	cudaMemcpy(h_C_, d_C_, vector_size_bytes, cudaMemcpyDeviceToHost); //Nos traemos los datos a CPU desde GPU

	//Comprobamos si nos da igual (CON UMBRAL de 1e-5)

	QueryPerformanceCounter(&t_ini);
	for (int i = 0; i < kNumElements; ++i) {
		if (fabs(h_A_[i] + h_B_[i] - h_C_[i] > 1e-5)) {
			std::cerr << "Error de verificacion en posicion " << i << ".";
			getchar(); exit(-1);
		}
	}
	QueryPerformanceCounter(&t_fin);
	secsH = performancecounter_diff(&t_fin, &t_ini);

	//PASO 6: LIBERAR MEMORIA Y DISPOSITIVO
	free(h_A_); free(h_B_); free(h_C_);
	cudaFree(d_A_); cudaFree(d_B_); cudaFree(d_C_);
	cudaDeviceReset();
	
	int numOperacionesPorThread = 1 + ((kNumElements - 1) / (threadsPerBlock * blocksPerGrid));
	std::cerr << "Operaciones realizadas correctamente.\n\n";
	printf("Hemos sumado vectores de %d posiciones, usando bloques de %d hilos (el maximo permitido por nuestra grafica)", kNumElements, threadsPerBlock);
	printf(" y grids de %d bloques (el tamanyo maximo de grid de nuestra grafica es de %d bloques), por lo tanto cada thread", blocksPerGrid, maxBlocksPerGrid);
	printf(" realizara un maximo de %d operacion(es).", numOperacionesPorThread);
	printf("\n\nSe tardaron %f segundos en ejecutar la operacion en el Host", secsH);
	getchar();

	return 0;
}