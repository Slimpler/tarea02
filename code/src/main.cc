#include <global.hh>
#include <algorithm>
#include <RandomUnifStream.hpp>
#include <Timing.hpp>
#include <MatrixToMem.hpp>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

/////////////////////////////////////////////////////////////////////////////////
//   Usage:
//           ./program_name  .......
//
//   Description:
//                ...................
//
/////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////
//  Sorting Network
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//
//  Input:
//      __m128i*  dataRegisters  : Arreglo de 4 vectores __m128i, cada uno
//                                con una secuencia desordenada de 4 enteros
//
//  Output:
//    La secuencia de 4 enteros  ordenada de cada vector se almacena
//    en las columnas del arreglo 'dataRegisters'.
//

void sortNet(__m128i* dataRegisters) {

	__m128i r_min1, r_max1, r_min2, r_max2, r_min3, r_max3, r_min4, r_max4, r_min5, r_max5;

	// Paso 1
	r_min1 = _mm_min_epi32(dataRegisters[0], dataRegisters[2]);
	r_max1 = _mm_max_epi32(dataRegisters[0], dataRegisters[2]);
	// Paso 2
	r_min2 = _mm_min_epi32(dataRegisters[1], dataRegisters[3]);
	r_max2 = _mm_max_epi32(dataRegisters[1], dataRegisters[3]);
	// Paso 3
	r_min3 = _mm_min_epi32(r_max1, r_max2);
	r_max3 = _mm_max_epi32(r_max1, r_max2);
	// Paso 4
	r_min4 = _mm_min_epi32(r_min1, r_min2);
	r_max4 = _mm_max_epi32(r_min1, r_min2);
	// Paso 5
	r_min5 = _mm_min_epi32(r_min3, r_max4);
	r_max5 = _mm_max_epi32(r_min3, r_max4);

	/* Salida:  */
	dataRegisters[0] = r_min4;
	dataRegisters[1] = r_min5;
	dataRegisters[2] = r_max5;
	dataRegisters[3] = r_max3;
}

//////////////////////////////////////////////////////////////////////
// transpose a matrix vector
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//   Input:
//       __m128i*  dataReg  : Arreglo de 4 vectores __m128i
//   Output:
//       __m128i*  dataReg  : Arreglo de 4 vectores que es 
//                            la matriz transpuesta de la original
//

void transpose(__m128i*  dataReg){
	__m128i S[4];

	S[0] = _mm_unpacklo_epi32(dataReg[0], dataReg[1]);
	S[1] = _mm_unpacklo_epi32(dataReg[2], dataReg[3]);
	S[2] = _mm_unpackhi_epi32(dataReg[0], dataReg[1]);
	S[3] = _mm_unpackhi_epi32(dataReg[2], dataReg[3]);

	dataReg[0] = _mm_unpacklo_epi64(S[0], S[1]);
	dataReg[1] = _mm_unpackhi_epi64(S[0], S[1]);
	dataReg[2] = _mm_unpacklo_epi64(S[2], S[3]);
	dataReg[3] = _mm_unpackhi_epi64(S[2], S[3]);
	
}

//////////////////////////////////////////////////////////////////////
//  Bitonic sorter
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//  Input:
//      __m128i*  dataReg1  : secuencia ordenada de 4 enteros ascedente
//      __m128i*  dataReg2  : secuencia ordenada de 4 enteros ascedente
//
//  Output:
//    La secuencia de 8 enteros totalmente ordenada se almacena en:
//      __m128i*  dataReg1   
//      __m128i*  dataReg2 
//

void bitonicSorter(__m128i*  dataReg1, __m128i*  dataReg2)
{
	
	//Reordenar dataReg2 para que la entrada sea una secuencia bitónica
	*dataReg2 = _mm_shuffle_epi32(*dataReg2, _MM_SHUFFLE(0, 1, 2, 3));
	
	//Código asociados a cada nivel del Bitonic Sorter
	/* Se obtiene el maximo y minimo de los registros
 	la variable m_aux es utilizada para alamacenar el minimo*/
	__m128i m_aux = _mm_min_epi32(*dataReg1, *dataReg2);
	*dataReg2 = _mm_max_epi32(*dataReg1, *dataReg2);
	*dataReg1 = m_aux;

	//Se extraen los datos
	uint32_t d_0 = _mm_extract_epi32(*dataReg1, 0);
    uint32_t d_1 = _mm_extract_epi32(*dataReg1, 1);
    uint32_t d_2 = _mm_extract_epi32(*dataReg1, 2);
    uint32_t d_3 = _mm_extract_epi32(*dataReg1, 3);
    uint32_t D_0 = _mm_extract_epi32(*dataReg2, 0);
    uint32_t D_1 = _mm_extract_epi32(*dataReg2, 1);
    uint32_t D_2 = _mm_extract_epi32(*dataReg2, 2);
    uint32_t D_3 = _mm_extract_epi32(*dataReg2, 3);

	//Se almacenan los datos maximos y minimos de cada registro
	*dataReg1=_mm_setr_epi32(d_0, D_0, d_2, D_2);
    *dataReg2=_mm_setr_epi32(d_1, D_1, d_3, D_3);
	
	*dataReg1 = _mm_unpackhi_epi64(*dataReg2, *dataReg1);
	*dataReg2 = _mm_unpacklo_epi64(*dataReg2, *dataReg1);

    //Se obtienen lo maximos y minimos
	m_aux = _mm_min_epi32(*dataReg1, *dataReg2);
	*dataReg2 = _mm_max_epi32(*dataReg1, *dataReg2);
	*dataReg1 = m_aux;

	//Se extraen los datos
	uint32_t r_0 = _mm_extract_epi32(*dataReg1, 0);
    uint32_t r_1 = _mm_extract_epi32(*dataReg1, 1);
    uint32_t r_2 = _mm_extract_epi32(*dataReg1, 2);
    uint32_t r_3 = _mm_extract_epi32(*dataReg1, 3);
    uint32_t R_0 = _mm_extract_epi32(*dataReg2, 0);
    uint32_t R_1 = _mm_extract_epi32(*dataReg2, 1);
    uint32_t R_2 = _mm_extract_epi32(*dataReg2, 2);
    uint32_t R_3 = _mm_extract_epi32(*dataReg2, 3);

    //Se almacenan los datos maximos y minimos de cada registro
	*dataReg1 = _mm_setr_epi32(r_0, R_0, r_2, R_2);
    *dataReg2 = _mm_setr_epi32(r_1, R_1, r_3, R_3);

    //Se obtienen los maximos y minimos
	m_aux = _mm_min_epi32(*dataReg1, *dataReg2);
	*dataReg2 = _mm_max_epi32(*dataReg1, *dataReg2);
	*dataReg1 = m_aux;

	//Ultima extraccion de datos para su utilización
	uint32_t n_0 = _mm_extract_epi32(*dataReg1, 0);
    uint32_t n_1 = _mm_extract_epi32(*dataReg1, 1);
    uint32_t n_2 = _mm_extract_epi32(*dataReg1, 2);
    uint32_t n_3 = _mm_extract_epi32(*dataReg1, 3);
    uint32_t N_0 = _mm_extract_epi32(*dataReg2, 0);
    uint32_t N_1 = _mm_extract_epi32(*dataReg2, 1);
    uint32_t N_2 = _mm_extract_epi32(*dataReg2, 2);
    uint32_t N_3 = _mm_extract_epi32(*dataReg2, 3);	

	//Salida de los registros ordenados
	*dataReg1=_mm_setr_epi32(n_0, N_0, n_2, N_2);
    *dataReg2=_mm_setr_epi32(n_1, N_1, n_3, N_3);
}

//////////////////////////////////////////////////////////////////////
//  Bitonic Merge Network
//      **********************************************************
//      ***Importante: el paso de parÃ¡metros es por referencia.***
//      **********************************************************
//  Input:
//       __m128i*  dataReg  : Arreglo de 4 vectores ordenados
//                            individualmente
//
//  Output:
//      __m128i*  dataReg  : Arreglo de 4 vectores ordenados 
//                           globalmente
//

void BNM(__m128i*  dataReg){
	bitonicSorter(&dataReg[0], &dataReg[1]);
	bitonicSorter(&dataReg[2], &dataReg[3]);
	bitonicSorter(&dataReg[1], &dataReg[2]);
	bitonicSorter(&dataReg[0], &dataReg[1]);
	bitonicSorter(&dataReg[2], &dataReg[3]);
}

void uso(std::string pname)
{
	std::cerr << "Uso: " << pname << " --fname MATRIX_FILE" << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{

	std::string fileName;
	
	//////////////////////////////////////////
	//  Read command-line parameters easy way
	if(argc != 3){
		uso(argv[0]);
	}
	std::string mystr;
	for (size_t i=0; i < argc; i++) {
		mystr=argv[i];
		if (mystr == "--fname") {
			fileName = argv[i+1];
		}
	}

	
	
	////////////////////////////////////////////////////////////////
	// Transferir la matriz del archivo fileName a memoria principal
	Timing timer0, timer1, timer2, timer3, timer4, timer_total;
	timer_total.start();
	//
	timer0.start();
	MatrixToMem m1(fileName);
	timer0.stop();
	//
	timer1.start();
	std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
	timer1.stop();
	//
	timer2.start();
	MatrixToMem m2(fileName);
	timer2.stop();

	//Comienza el Ordenamiento Vectorial
	timer3.start();
	__m128i  dataReg[4];
	for(size_t i = 0; i < m2._nfil; i += 16){
		if(m2._nfil==1000 && i==992){
			dataReg[0] = _mm_setr_epi32(m2._matrixInMemory[i],m2._matrixInMemory[i+1],m2._matrixInMemory[i+2],m2._matrixInMemory[i+3]);
			dataReg[1] = _mm_setr_epi32(m2._matrixInMemory[i+4],m2._matrixInMemory[i+5],m2._matrixInMemory[i+6],m2._matrixInMemory[i+7]);
			dataReg[2] = _mm_setr_epi32(m2._matrixInMemory[i+8],m2._matrixInMemory[i+9],m2._matrixInMemory[i+10],m2._matrixInMemory[i+11]);
			dataReg[3] = _mm_setr_epi32(m2._matrixInMemory[i+12],m2._matrixInMemory[i+13],m2._matrixInMemory[i+14],m2._matrixInMemory[i+15]);
			//Ordenando los 4 datos de cada registro a con Sorting Network
			sortNet(dataReg);
			transpose(dataReg);
			//Ordenando los 8 datos en total de dos registros con Bitonic Sorter
			bitonicSorter(&dataReg[0], &dataReg[1]);
			m2._matrixInMemory[i] = _mm_extract_epi32(dataReg[0],0);
			m2._matrixInMemory[i+1] = _mm_extract_epi32(dataReg[0],1);
			m2._matrixInMemory[i+2] = _mm_extract_epi32(dataReg[0],2);
			m2._matrixInMemory[i+3] = _mm_extract_epi32(dataReg[0],3);
			m2._matrixInMemory[i+4] = _mm_extract_epi32(dataReg[1],0);
			m2._matrixInMemory[i+5] = _mm_extract_epi32(dataReg[1],1);
			m2._matrixInMemory[i+6] = _mm_extract_epi32(dataReg[1],2);
			m2._matrixInMemory[i+7] = _mm_extract_epi32(dataReg[1],3);
			break;
		}

		dataReg[0] = _mm_setr_epi32(m2._matrixInMemory[i],m2._matrixInMemory[i+1],m2._matrixInMemory[i+2],m2._matrixInMemory[i+3]);
		dataReg[1] = _mm_setr_epi32(m2._matrixInMemory[i+4],m2._matrixInMemory[i+5],m2._matrixInMemory[i+6],m2._matrixInMemory[i+7]);
		dataReg[2] = _mm_setr_epi32(m2._matrixInMemory[i+8],m2._matrixInMemory[i+9],m2._matrixInMemory[i+10],m2._matrixInMemory[i+11]);
		dataReg[3] = _mm_setr_epi32(m2._matrixInMemory[i+12],m2._matrixInMemory[i+13],m2._matrixInMemory[i+14],m2._matrixInMemory[i+15]);
		//Ordenabdo los 4 datos de cada registro a traves del Sorting Network
		sortNet(dataReg);
		transpose(dataReg);
		//Ordenando los 8 datos en total de dos registros con Bitonic Sorter
		bitonicSorter(&dataReg[0], &dataReg[1]);
		bitonicSorter(&dataReg[2], &dataReg[3]);
		//Ordenando los 16 datos con Bitonic Merge Network
		BNM(dataReg);
		transpose(dataReg);
		//Copiando el contenido de los registros vectoriales a la memoria principal
		m2._matrixInMemory[i] = _mm_extract_epi32(dataReg[0],0);
		m2._matrixInMemory[i+1] = _mm_extract_epi32(dataReg[0],1);
		m2._matrixInMemory[i+2] = _mm_extract_epi32(dataReg[0],2);
		m2._matrixInMemory[i+3] = _mm_extract_epi32(dataReg[0],3);
		m2._matrixInMemory[i+4] = _mm_extract_epi32(dataReg[1],0);
		m2._matrixInMemory[i+5] = _mm_extract_epi32(dataReg[1],1);
		m2._matrixInMemory[i+6] = _mm_extract_epi32(dataReg[1],2);
		m2._matrixInMemory[i+7] = _mm_extract_epi32(dataReg[1],3);
		m2._matrixInMemory[i+8] = _mm_extract_epi32(dataReg[2],0);
		m2._matrixInMemory[i+9] = _mm_extract_epi32(dataReg[2],1);
		m2._matrixInMemory[i+10] = _mm_extract_epi32(dataReg[2],2);
		m2._matrixInMemory[i+11] = _mm_extract_epi32(dataReg[2],3);
		m2._matrixInMemory[i+12] = _mm_extract_epi32(dataReg[3],0);
		m2._matrixInMemory[i+13] = _mm_extract_epi32(dataReg[3],1);
		m2._matrixInMemory[i+14] = _mm_extract_epi32(dataReg[3],2);
		m2._matrixInMemory[i+15] = _mm_extract_epi32(dataReg[3],3);
	}
	timer3.stop();
	timer4.start();
	std::sort(m2._matrixInMemory, m2._matrixInMemory + m2._nfil);
	timer4.stop();
	timer_total.stop();

	//Mostrando en Pantalla
	std::cout << "- Time to transfer to main memory: " << timer0.elapsed() << std::endl;
	std::cout << "- Time to sort in main memory(m1): " << timer1.elapsed() << std::endl;
	std::cout << "- Time to transfer to main memory: " << timer2.elapsed() << std::endl;
	std::cout << "- Tiempo de ordenamiento vectorial "<< timer3.elapsed() <<std::endl;
	std::cout << "- Time to sort in main memory (m2): " << timer4.elapsed() << std::endl;
	std::cout << "- Tiempo ejecución total: " << timer_total.elapsed() << std::endl;


	return(EXIT_SUCCESS);
}

