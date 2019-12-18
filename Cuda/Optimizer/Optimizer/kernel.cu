#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include "file_reader.h"
#include "bit_converter.h"
#include <iostream>

using namespace std;


#define BLOCK_SIZE 1024
#define DIVERGENCE_CONTROL_CYCLE 128 //Через какое кол-во операций мы проверяем на необходимость ужатия, чтобы избежать дивергенции?
#define DIVERGENCE_CONTROL_THRESHOLD 32 //Если какое число потоков простаивают надо ужимать? Думаю это значение вряд ли когда-нибудь изменится.
#define REGISTRY_STRUCTURE_LENGTH 12
#define SIZING_STRUCTURE_LENGTH 4
#define DATA_STRUCTURE_LENGTH 4
#define RESERVED_DATA_LENGHT 2

#define MAX_FLAGS_LENGTH_FOR_SPRITE 8192 //Для 256х256 спрайта он именно такой

__constant__ short SizingsCount;
__constant__ short SpritesCount;
__constant__ int ByteLineLength;
__constant__ int BitLineLength;

__constant__ short SizingWidths[22];
__constant__ short SizingHeights[22];
__constant__ int SpriteByteOffsets[654];
__constant__ int SpriteBitOffsets[654];
__constant__ short SpriteWidths[654];
__constant__ short SpriteHeights[654];
__constant__ int VoidOffsets[14388];

__global__ void mainKernel(unsigned char* rgbaData, unsigned char* voids, unsigned char* rgbaFlags, unsigned int* workingOffsets, unsigned int* results)
{
	/*if (threadIdx.x == 0)
		printf("x = %d, z = %d\n", blockIdx.x, blockIdx.z);
	return;*/

	//int ourSpriteIndex = blockIdx.x;
	//int candidateSpriteIndex = candidateIndex;
	//int sizingIndex = blockIdx.z;

	//int ourByteOffset = SpriteByteOffsets[blockIdx.x];
	//int ourBitOffset = SpriteBitOffsets[blockIdx.x];
	//short ourWidth = SpriteWidths[blockIdx.x];
	//short ourHeight = SpriteHeights[blockIdx.x];
	//int ourSquare = SpriteWidths[blockIdx.x] * SpriteHeights[blockIdx.x];
	int ourBitsSquare = (SpriteWidths[blockIdx.x] * SpriteHeights[blockIdx.x]) / 8;
	if ((SpriteWidths[blockIdx.x] * SpriteHeights[blockIdx.x]) % 8 != 0)
		ourBitsSquare++;

	//short sizingWidth = SizingWidths[blockIdx.z];
	//short sizingHeight = SizingHeights[blockIdx.z];

	//__shared__ char ourRFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	//__shared__ char ourGFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	//__shared__ char candidateRFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	//__shared__ char candidateGFlags[MAX_FLAGS_LENGTH_FOR_SPRITE];
	//__shared__ char candidateVoidMap[MAX_FLAGS_LENGTH_FOR_SPRITE];//Экономим регистры
	__shared__ char cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 5];

	int numberOfTimesWeNeedToLoadSelf = ourBitsSquare / BLOCK_SIZE;
	if (ourBitsSquare % BLOCK_SIZE != 0)
		numberOfTimesWeNeedToLoadSelf++;

	for (size_t i = 0; i < numberOfTimesWeNeedToLoadSelf; i++)
	{
		int byteAddress = i * BLOCK_SIZE + threadIdx.x;
		if (byteAddress >= ourBitsSquare)
			continue;
		cachedBits[byteAddress] = rgbaFlags[SpriteBitOffsets[blockIdx.x] + byteAddress];
		cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE + byteAddress] = rgbaFlags[BitLineLength + SpriteBitOffsets[blockIdx.x] + byteAddress];
	}

	int ourWorkingHeight = SpriteHeights[blockIdx.x] - SizingHeights[blockIdx.z];
	int ourWorkingSquare = (SpriteWidths[blockIdx.x] - SizingWidths[blockIdx.z]) * ourWorkingHeight;
	int numberOfTasksPerThread = ourWorkingSquare / BLOCK_SIZE;
	if (ourWorkingSquare % BLOCK_SIZE != 0)
		numberOfTasksPerThread++;

	for (size_t candidateIndex = 0; candidateIndex < SpritesCount; candidateIndex++)
	{
		/*int candidateByteOffset = SpriteByteOffsets[candidateIndex];
		int candidateBitOffset = SpriteBitOffsets[candidateIndex];*/
		//int candidateWidth = SpriteWidths[candidateIndex];
		//int candidateHeight = SpriteHeights[candidateIndex];
		//int candidateSquare = SpriteWidths[candidateIndex] * SpriteHeights[candidateIndex];
		int candidateBitsSquare = (SpriteWidths[candidateIndex] * SpriteHeights[candidateIndex]) / 8;
		if ((SpriteWidths[candidateIndex] * SpriteHeights[candidateIndex]) % 8 != 0)
			candidateBitsSquare++;

		int numberOfTimesWeNeedToLoadCandidate = candidateBitsSquare / BLOCK_SIZE;
		if (candidateBitsSquare % BLOCK_SIZE != 0)
			numberOfTimesWeNeedToLoadCandidate++;

		//if (blockIdx.x == 7 && candidateIndex == 7 && blockIdx.z == 18) //Так мы обойдемся без повторов, только 1 блок будет логировать
		//	printf("rgbaFlags[%d] = %d\n", threadIdx.x, rgbaFlags[threadIdx.x]);
		//printf("BitLineLength = %d, ByteLineLength = %d, SpritesCount = %d, SizingsCount = %d\n", BitLineLength, ByteLineLength, SpritesCount, SizingsCount);

		//__syncthreads();

		for (size_t i = 0; i < numberOfTimesWeNeedToLoadCandidate; i++)
		{
			int byteAddress = i * BLOCK_SIZE + threadIdx.x;
			if (byteAddress >= candidateBitsSquare)
				continue;
			cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 2 + byteAddress] = rgbaFlags[SpriteBitOffsets[candidateIndex] + byteAddress];
			cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 3 + byteAddress] = rgbaFlags[BitLineLength + SpriteBitOffsets[candidateIndex] + byteAddress];
		}

		//int candidateWidthMinusSizing = SpriteWidths[candidateIndex] - SizingWidths[blockIdx.z];
		//int candidateHeightMinusSizing = SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z];
		//int candidateVoidAreaSquare = (SpriteWidths[candidateIndex] - SizingWidths[blockIdx.z]) * (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]);
		int candidateVoidAreaBitSquare = (SpriteWidths[candidateIndex] - SizingWidths[blockIdx.z]) * (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]) / 8;
		if ((SpriteWidths[candidateIndex] - SizingWidths[blockIdx.z]) * (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]) % 8 != 0)
			candidateVoidAreaBitSquare++;

		int numberOfTimesWeNeedToLoadVoid = candidateVoidAreaBitSquare / BLOCK_SIZE;
		if (candidateVoidAreaBitSquare % BLOCK_SIZE != 0)
			numberOfTimesWeNeedToLoadVoid++;

		int candidateVoidMapOffset = VoidOffsets[candidateIndex * SizingsCount + blockIdx.z];
		unsigned char* candidateVoidMapGlobal = voids + candidateVoidMapOffset;
		//if (blockIdx.x == 7 && candidateIndex == 7 && blockIdx.z == 18 && threadIdx.x < candidateVoidAreaSquare)
		//{
		//	int candidateX = threadIdx.x / (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]);
		//	int candidateY = threadIdx.x % (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]);
		//	printf("	void (%d, %d): %d\n", candidateX, candidateY, candidateVoidMapGlobal[threadIdx.x / 8] >> threadIdx.x % 8 & 1);
		//} //Проверили правильность апрсинга войдмап


		for (size_t i = 0; i < numberOfTimesWeNeedToLoadVoid; i++)
		{
			int voidByteAddress = i * BLOCK_SIZE + threadIdx.x;
			if (voidByteAddress >= candidateVoidAreaBitSquare)
				continue;
			cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 4 + voidByteAddress] = candidateVoidMapGlobal[voidByteAddress];
			//candidateVoidMap[voidByteAddress] = voids[SpriteBitOffsets[candidateIndex]];
		}

		__syncthreads(); //Обязательна синхронизация для того, чтобы потоки, которые не выполняли загрузку в шаред-память, не начали с этой шаред памятью работать, пока другие в нее еще не все загрузили, ибо результат - непредсказуем.

		////Проверяем, что все скопировалось правильно. Для этого выбираем случайный спрайт и логируем его флаги. Пускай будет спрайт №7
		//if (blockIdx.x == 7 && blockIdx.z == 18) //Так мы обойдемся без повторов, только 1 блок будет логировать
		//{
		//	if (threadIdx.x < SpriteWidths[blockIdx.x] * SpriteHeights[blockIdx.x])
		//	{
		//		int x = threadIdx.x / BLOCK_SIZE;
		//		int y = threadIdx.x % BLOCK_SIZE;
		//		printf("for pixel #%d (%d, %d) the flags of r and g are (%d, %d) == (%d, %d)\n", threadIdx.x, x, y, (cachedBits[threadIdx.x / 8] >> (threadIdx.x % 8)) & 1, (cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE + threadIdx.x / 8] >> (threadIdx.x % 8)) & 1, (cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 2 + threadIdx.x / 8] >> (threadIdx.x % 8)) & 1, (cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 3 + threadIdx.x / 8] >> (threadIdx.x % 8)) & 1);
		//	}
		//} //Проверил, работает


		//if (blockIdx.x == 7 && blockIdx.z == 18 && threadIdx.x < (SpriteWidths[candidateIndex] - SizingWidths[blockIdx.z]) * (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]))
		//{
		//	int candidateX = threadIdx.x / (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]);
		//	int candidateY = threadIdx.x % (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]);
		//	printf("	void (%d, %d): %d\n", candidateX, candidateY, cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 4 + threadIdx.x / 8] >> threadIdx.x % 8 & 1);
		//} //Проверили правильность парсинга войдмап
		//return;

		/*if (threadIdx.x == 0)
			printf("candidateIndex = %d, blocx = %d\n", candidateIndex, blockIdx.x);
		return;*/
		/*if (threadIdx.x == 0)
			printf("candidateIndex = %d, blocx = %d\n", blockIdx.x + candidateIndex, blockIdx.x);
		return;*/

		for (size_t taskIndex = 0; taskIndex < numberOfTasksPerThread; taskIndex++)
		{
			int ourWorkingPixelIndex = taskIndex * BLOCK_SIZE + threadIdx.x;
			if (ourWorkingPixelIndex >= ourWorkingSquare)
				break;

			int ourX = ourWorkingPixelIndex / ourWorkingHeight;
			int ourY = ourWorkingPixelIndex % ourWorkingHeight;
			//int coincidences = 0; //Значения меньше 0 - повторы

			/*
				Ок, тут мы имеем доступ к ourX/ourY координатам нашего спрайта и как бы предполагается, что мы будем работать с текущим кандидатом (candidateIndex).
				Предполагается, что мы пройдемся по кандидату и посчитаем общий счет для ourX/ourY/sizing-области на данном кандидате.
				Самое минимальное, что мы можем сделать - например сложить ourX/ourY/sizing с каждым первым пикселем кандидата.
				Нет, наверное еще минимальнее - посчитать сейчас войдмапы и вернуться после первой итерации. Будет легко проверить.
			*/

			//Считаем войдмапы.
			unsigned int voidness = 0;
			for (size_t x = 0; x < SizingWidths[blockIdx.z]; x++)
			{
				for (size_t y = 0; y < SizingHeights[blockIdx.z]; y++)
				{
					int ourPixelIndex = (ourX + x) * SpriteHeights[blockIdx.x] + ourY + y;
					if (rgbaData[ByteLineLength * 3 + SpriteByteOffsets[blockIdx.x] + ourPixelIndex] != 0)
					{
						voidness = 1;
						break;
					}
				}
				if (voidness != 0)
					break;
			}

			//printf("workingOffsets[%d] = %d\n", blockIdx.x, workingOffsets[blockIdx.x]);
			//return;
			results[(workingOffsets[blockIdx.x * SizingsCount + blockIdx.z] + ourX * ourWorkingHeight + ourY)/* * sizeof(int)*/] = voidness;
			//results[(workingOffsets[blockIdx.x] + ourX * ourWorkingHeight + ourY) * sizeof(int)] = voidness;
		}

		//Возвращаемся после первой итерации.

		//printf("asjdisajiod\n");
		return;

		//for (size_t taskIndex = 0; taskIndex < numberOfTasksPerThread; taskIndex++)
		//{
		//	int ourWorkingPixelIndex = taskIndex * BLOCK_SIZE + threadIdx.x;
		//	if (ourWorkingPixelIndex >= ourWorkingSquare)
		//		break;

		//	int ourX = ourWorkingPixelIndex / ourWorkingHeight;
		//	int ourY = ourWorkingPixelIndex % ourWorkingHeight;
		//	int coincidences = 0; //Значения меньше 0 - повторы


		//	for (size_t x = 0; x < SpriteWidths[candidateIndex] - SizingWidths[blockIdx.z]; x++)
		//	{
		//		if (coincidences < 0)
		//			break;

		//		for (size_t y = 0; y < SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]; y++)
		//		{
		//			if (coincidences < 0)
		//				break;

		//			int voidMapIndex = x * (SpriteHeights[candidateIndex] - SizingHeights[blockIdx.z]) + y;
		//			if ((cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 4 + voidMapIndex] >> (voidMapIndex % 8)) & 1 == 0) //Пустота
		//				continue;

		//			bool isTheSame = true;
		//			for (size_t xx = 0; xx < SizingWidths[blockIdx.z]; xx++)
		//			{
		//				for (size_t yy = 0; yy < SizingHeights[blockIdx.z]; yy++)
		//				{
		//					int ourPixelIndex = (ourX + xx) * SpriteHeights[blockIdx.x] + ourY + yy;
		//					int candidatePixelIndex = (x + xx) * SpriteHeights[candidateIndex] + y + yy;

		//					if ((cachedBits[ourPixelIndex / 8] >> (ourPixelIndex % 8)) & 1 != (cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 2 + candidatePixelIndex / 8] >> (candidatePixelIndex % 8)) & 1)
		//					{
		//						isTheSame = false;
		//						break;
		//					}

		//					if ((cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE + ourPixelIndex / 8] >> (ourPixelIndex % 8)) & 1 != (cachedBits[MAX_FLAGS_LENGTH_FOR_SPRITE * 3 + candidatePixelIndex / 8] >> (candidatePixelIndex % 8)) & 1)
		//					{
		//						isTheSame = false;
		//						break;
		//					}

		//					if (rgbaData[ByteLineLength * 3 + SpriteByteOffsets[blockIdx.x] + ourPixelIndex] != rgbaData[ByteLineLength * 3 + SpriteByteOffsets[candidateIndex] + candidatePixelIndex])
		//					{
		//						isTheSame = false;
		//						break;
		//					}

		//					//Если у нас пиксель прозрачный нас не интересуют остальные каналы
		//					if (rgbaData[ByteLineLength * 3 + SpriteByteOffsets[blockIdx.x] + ourPixelIndex] != 0)
		//					{
		//						if (rgbaData[SpriteByteOffsets[blockIdx.x] + ourPixelIndex] != rgbaData[SpriteByteOffsets[candidateIndex] + candidatePixelIndex])
		//						{
		//							isTheSame = false;
		//							break;
		//						}

		//						if (rgbaData[ByteLineLength + SpriteByteOffsets[blockIdx.x] + ourPixelIndex] != rgbaData[ByteLineLength + SpriteByteOffsets[candidateIndex] + candidatePixelIndex])
		//						{
		//							isTheSame = false;
		//							break;
		//						}

		//						if (rgbaData[ByteLineLength * 2 + SpriteByteOffsets[blockIdx.x] + ourPixelIndex] != rgbaData[ByteLineLength * 2 + SpriteByteOffsets[candidateIndex] + candidatePixelIndex])
		//						{
		//							isTheSame = false;
		//							break;
		//						}
		//					}
		//				}

		//				if (!isTheSame)
		//					break;
		//			}

		//			if (isTheSame)
		//			{
		//				if (blockIdx.x == candidateIndex && ourX * SpriteHeights[blockIdx.x] + ourY > x* SpriteHeights[candidateIndex] + y) //Если мы обнаружили область на том же спрайте, стоящую до нас - повтор
		//					coincidences = -1;
		//				else
		//					coincidences++;
		//			}
		//		}
		//	}

		//	results[workingOffsets[blockIdx.x] * sizeof(int) + ourX * ourWorkingHeight + ourY] += coincidences;

		//	//if (blockIdx.z == 0)
		//	printf("x = %d, z = %d, coincidences = %d\n", blockIdx.x, blockIdx.z, coincidences);
		//	//printf("coincidences = %d\n", coincidences);
		//	return;
		//}
	}
}

int main()
{
    string path = "P:\\U\\Some2DGame\\Cuda\\info\\data.bytes";
    tuple<char*, int> blobTuple = file_reader::readFile(path);
    char* blob = get<0>(blobTuple);
    int blobLength = get<1>(blobTuple);

    //blob состоит из мета-инфы, которую мы не используем в рассчетах, и основной инфы. Мета - 4 байта длина и собственно данные.
    int metaLength = bit_converter::GetInt(blob, 0);
    int combinedDataOffset = metaLength + sizeof(int); //указатель на начало массива основных данных

    short spritesCount = bit_converter::GetShort(blob, combinedDataOffset + RESERVED_DATA_LENGHT); // первые 2 байта основных данных зарезервированы, вторые - кол-во спрайтов.
    cudaMemcpyToSymbol(SpritesCount, &spritesCount, sizeof(short)); //Сразу записываем их в константы устройства
    short sizingsCount = bit_converter::GetShort(blob, combinedDataOffset + RESERVED_DATA_LENGHT + sizeof(short)); //третьи 2 байта - кол-во сайзингов
    cudaMemcpyToSymbol(SizingsCount, &sizingsCount, sizeof(short)); //Тоже записываем сразу туда.

    //Определяем массивы и длины данных сайзингов и регистра

    //Сначала сайзинги...
    char* sizingsBlob = blob + combinedDataOffset + RESERVED_DATA_LENGHT + sizeof(short) * 2;
    int sizingsBlobLenght = sizingsCount * SIZING_STRUCTURE_LENGTH; //Сайзинги состоят из 2 шортов - х и у
    //Записываем сайзинги на девайс. Они там идут последовательно, сначала иксы потом игрики
    int sizingsLineLength = sizeof(short) * sizingsCount;
    short* sizingWidths = (short*)sizingsBlob;
    short* sizingHeights = (short*)(sizingsBlob + sizingsLineLength);
    cudaMemcpyToSymbol(SizingWidths, sizingWidths, sizingsLineLength);
    cudaMemcpyToSymbol(SizingHeights, sizingHeights, sizingsLineLength);


	char* registryBlob = sizingsBlob + sizingsBlobLenght;
	int registryBlobLength = spritesCount * REGISTRY_STRUCTURE_LENGTH; //регистр на данный момент состоит из 2 шортов и 2 интов, длина структуры задается через REGISTRY_STRUCTURE_LENGTH
	//Записываем регистр на девайс. Они там идут последовательно, сначала байтовые оффсеты потом битовые, потом иксы, потом игрики
	int registryLineCount = spritesCount * sizingsCount;
	int* spriteByteOffsets = (int*)registryBlob;
	int* spriteBitOffsets = (int*)(registryBlob + spritesCount * sizeof(int));
	short* spriteWidths = (short*)(registryBlob + spritesCount * sizeof(int) * 2);
	short* spriteHeights = (short*)(registryBlob + spritesCount * (sizeof(int) * 2 + sizeof(short)));
	cudaMemcpyToSymbol(SpriteByteOffsets, spriteByteOffsets, spritesCount * sizeof(int));
	cudaMemcpyToSymbol(SpriteBitOffsets, spriteBitOffsets, spritesCount * sizeof(int));
	cudaMemcpyToSymbol(SpriteWidths, spriteWidths, spritesCount * sizeof(short));
	cudaMemcpyToSymbol(SpriteHeights, spriteHeights, spritesCount * sizeof(short));

	//Дальше идет длина 1 канала цвета
	int byteLineLength = bit_converter::GetInt(registryBlob + registryBlobLength, 0);
	cudaMemcpyToSymbol(ByteLineLength, &byteLineLength, sizeof(int)); //Сразу записываем ее в константы

	//Дальше идут данные. А зная длину 1 канала цвета, мы можем легко посчитать общую длину массива цветов
	char* rgbaBlob = registryBlob + registryBlobLength + sizeof(int);
	int rgbaBlobLength = byteLineLength * DATA_STRUCTURE_LENGTH; //Длина структуры основных данных у нас 4 - по 1 байту на канал.
	//Сразу записываем их в глобальную память
	char* deviceRgbaDataPtr;
	cudaMalloc((void**)&deviceRgbaDataPtr, rgbaBlobLength);
	cudaMemcpy(deviceRgbaDataPtr, rgbaBlob, rgbaBlobLength, cudaMemcpyHostToDevice);

	/*
		Это были основные данные. Дальше идут вспомогательные.
	*/


	//Следующими за данными о цвете идут оффсеты карт пустот
	int voidMapsCount = spritesCount * sizingsCount; //Карта пустот есть отдельно для каждого спрайта каждого сайзинга
	int* voidMapsOffsets = (int*)(rgbaBlob + rgbaBlobLength);
	int voidMapsOffsetsLength = voidMapsCount * sizeof(int);
	cudaMemcpyToSymbol(VoidOffsets, voidMapsOffsets, voidMapsOffsetsLength); //Пишем ее в константы

	//Потом длина уже непосредственно карт и за ней - сами данные карты
	int voidsBlobLength = bit_converter::GetInt((char*)(voidMapsOffsets + voidMapsCount), 0);
	char* voidsBlob = (char*)(voidMapsOffsets + voidMapsCount) + sizeof(int);
	//Пишем эти данные в глобал мемори
	char* deviceVoidsPtr;
	cudaMalloc((void**)&deviceVoidsPtr, voidsBlobLength);
	cudaMemcpy(deviceVoidsPtr, voidsBlob, voidsBlobLength, cudaMemcpyHostToDevice);

	//Дальше длина канала основных данных в битах
	int bitLineLength = bit_converter::GetInt(voidsBlob + voidsBlobLength, 0);
	cudaMemcpyToSymbol(BitLineLength, &bitLineLength, sizeof(int)); // Сразу записываем ее
	char* rgbaFlags = voidsBlob + voidsBlobLength + sizeof(int);
	int rgbaFlagsLength = bitLineLength * DATA_STRUCTURE_LENGTH;
	//Пишем их тоже в глобальную
	char* deviceRgbaFlagsPtr;
	cudaMalloc((void**)&deviceRgbaFlagsPtr, rgbaFlagsLength);
	cudaMemcpy(deviceRgbaFlagsPtr, rgbaFlags, rgbaFlagsLength, cudaMemcpyHostToDevice);

	/*
		Ок, нам нужны области памяти для хранения промежуточных результатов. Первый промежуточный результат - это счет спрайто-размеро-спрайта. Второй - общий счет спрайто-размера. Первый получается довольно
		огромным. Допустим, у нас 3000 спрайтов 256х256, тогда размер структуры для первого промежуточного результата нужен будет такой: 256х256х3000х22х3000х4 байт, то есть... всего лишь 2 терабайта памяти.
		Прикольно. Ок. Я посчитал, если не брать в расчет первую промежуточную структуру, а сразу писать во вторую - максимальное кол-во спрайтов размера 256х256, которое мы сможем взять - 1733. Т.е. где-то
		113 мегапикселей за раз. Ну, это, конечно, не мало само по себе, но с другой - это всего лишь 7 4к-текстур. Конечно, можно будет как-то выкручиваться - свопом, уменьшением кол-ва сайзингов, но вообще
		в будущем, конечно, стоит придумать нормальное решение для этой проблемы. Может обсчитывать в несколько проходов с выгрузкой промежуточных результатов в оперативку. В принципе не так сложно должно
		быть сделать. А на первое время, я думаю, такого кол-ва хватит. А стратегию подсчета конечно же надо будет поменять на подсчет сразу в итоговый результат. Т.е. пропускаем первую промежуточную стадию.
	*/

	//Размерность матриц результатов у нас не совпадает с размерностью спрайтов из-за сайзингов.
	//Еще нам нужны байтовые оффсеты рабочих областей спрайтов, раз мы хотим экономить место. Битовые войд-оффсеты не подходят, т.к. там округляются значения до кратных 8.
	//В будущем вот это вот всё надо будет перенести на клиента.
	unsigned int resultsCount = 0;
	unsigned int* workingSpriteOffsets = (unsigned int*)malloc(sizingsCount * spritesCount * sizeof(int));
	unsigned int currentOffset = 0;
	for (size_t i = 0; i < spritesCount; i++)
	{
		int spriteWidth = spriteWidths[i];
		int spriteHeight = spriteHeights[i];
		for (size_t j = 0; j < sizingsCount; j++)
		{
			short sizingWidth = sizingWidths[j];
			short sizingHeight = sizingHeights[j];

			unsigned int currentWorkingSpriteLength = (spriteWidth - sizingWidth) * (spriteHeight - sizingHeight);
			resultsCount += currentWorkingSpriteLength;
			//std::cout << "spriteWidth = " << spriteWidth << " sizingWidth = " << sizingWidth << " spriteHeight = " << spriteHeight << " sizingHeight = " << sizingHeight << "\n";
			workingSpriteOffsets[i * sizingsCount + j] = currentOffset;
			currentOffset += currentWorkingSpriteLength;
		}
	}
	char* deviceResultsPtr;
	cudaMalloc((void**)&deviceResultsPtr, resultsCount * sizeof(int));
	cudaMemset(deviceResultsPtr, 0, resultsCount * sizeof(int));
	unsigned int* deviceWorkingSpriteOffsetsPtr;
	cudaMalloc((void**)&deviceWorkingSpriteOffsetsPtr, sizingsCount * spritesCount * sizeof(unsigned int));
	cudaMemcpy(deviceWorkingSpriteOffsetsPtr, workingSpriteOffsets, sizingsCount * spritesCount * sizeof(unsigned int), cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE);
	dim3 grid(spritesCount, 1, sizingsCount); //Сайзингов будет меньше, чем спрайтов, так что сайзинги записываем в z
	mainKernel << <grid, block >> > ((unsigned char*)deviceRgbaDataPtr, (unsigned char*)deviceVoidsPtr, (unsigned char*)deviceRgbaFlagsPtr, deviceWorkingSpriteOffsetsPtr, (unsigned int*)deviceResultsPtr);

	cudaDeviceSynchronize();

	//testing...
	int* gpuResults = (int*)malloc(resultsCount * sizeof(int));
	cudaMemcpy(gpuResults, deviceResultsPtr, resultsCount * sizeof(int), cudaMemcpyDeviceToHost);
	int spriteTestIndex = 0;
	int sizingTestIndex = 18;

	int testOffsetIndex = spriteTestIndex * sizingsCount + sizingTestIndex;
	int voidOffset = voidMapsOffsets[testOffsetIndex];
	int resultOffset = workingSpriteOffsets[testOffsetIndex];

	short testSpriteWidth = spriteWidths[spriteTestIndex];
	short testSpriteHeight = spriteHeights[spriteTestIndex];
	short testSizingWidth = sizingWidths[sizingTestIndex];
	short testSizingHeight = sizingHeights[sizingTestIndex];
	short workingWidth = testSpriteWidth - testSizingWidth;
	short workingHeight = testSpriteHeight - testSizingHeight;

	int testCount = 240;
	std::cout << "First " << testCount << " voids:\n";
	for (size_t i = 0; i < testCount; i++)
	{
		int x = i / workingHeight;
		int y = i % workingHeight;
		std::cout << i << " (" << x << "," << y << "): voidsBlob = " << ((voidsBlob[voidOffset + i / 8] >> (i % 8)) & 1) << ", results = " << gpuResults[resultOffset + i] << "\n";
	}

	//testing...

	cudaFree(deviceRgbaDataPtr);
	cudaFree(deviceVoidsPtr);
	cudaFree(deviceRgbaFlagsPtr);
	cudaFree(deviceResultsPtr);
	cudaFree(deviceWorkingSpriteOffsetsPtr);

	free(workingSpriteOffsets);
	free(gpuResults);
	free(blob);


    printf("ok\n");

    return 0;
}
