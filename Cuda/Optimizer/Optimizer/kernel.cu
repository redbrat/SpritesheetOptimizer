
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include "file_reader.h"
#include "bit_converter.h"

using namespace std;

/*
Ок, это первый метод, который я параллелю, поэтому давайте решим, как я это буду делать. Просто по индексу блока мы не можем вычислить никакие индексы,
потому что ширина-высота спрайтов неодинакова. Точнее можем, но это будет очень неэффективно. А что мы должны сделать - это каким-то образом разделить
задачи по спрайтам так же как по сайзингам и инфу о сайзингах и размерах спрайтов закэшировать. Можно сказать так - кол-во блоков = кол-во сайзингов * 
кол-во спрайтов. Таким образом, поток может легко вычислить спрайт и сайзинг, которые ему обрабатывать. Если его индекс 126, а кол-во сайзингов и спрайтов
 - 22 и 40 соответственно, то его сайзинг = 126 / 40, а спрайт - 126 % 40, т.е. сайзинг четвертый, спрайт - 6й. А всего блоков 22 * 40 = 880. Че-то мало. 
Хотя, это только блоки. А как внутри потоки будут узнавать о своих конкретных назначенных пикспелях? А вот как. Мы будем запускать кернел для каждого 
из 880 вариантов. Поэтому для каждого потока будет очевидно кто он и где он, т.к. будет передваваться ширина и высота. Иммет ли смысл делать столько 
кернелов? Для среднего размера спрайтов, скажем 128х60 вся работа целого кернела будет состоять из всего нескольких сотен тысяч операций. Целого кернела, 
Карл! Это если размер блока задать 128, то будет всего 8 блоков. И это для средних размеров спрайтов. Не маловато ли работы? Может быть имеет смысл сделать 
опциональное разделение кернелов по спрайтам - 1 кернел == 1 спрайт. Тогда если размер блока будет 128, блоков будет уже по крайней мере в районе сотни. 
Внутри такого блока мы на вход получаем смещение спрайта, ширину и высоту, а, будучи потоком, мы можем вычислить свою позицию стандартно, т.к. ширина и 
высота тут постоянные. Возможно для уменьшения дайверженс и лучшей работы с памятью стоит изолировать также разные сайзинги по блокам. Т.е. если размер 
блока будет, скажем, 128, сайзинг - 8х8, а спрайт 128х48, мы запускаем кернел, передавая ему оффсет спрайта в данных, 128 и 48. На уровне блоков мы действуем 
следующим образом. У нас сайзинг 8х8, поэтому рабочая ширина у нас 120, а высота - 40. Таким образом нам нужно иметь 120 х 40 = 4800 потоков. Таким образом 
на этом сайзинге у нас будут работать 4800 / 128 + 1 = 38 блоков. В статической памяти у них будет храниться вся нужная им инфа - их сайзинг, их референсная 
область, возможно, если памяти хватит - их карты пустоты для данного спрайта. Вообще-то памяти как-то мало везде... Может использовать 1 блок на 
сайзинг-спрайт, просто очень большой? Скажем, пусть размер блока будет максимальным (1024), тогда мы сможем грузить в шаренную память спрайт, если не 
целиком, то координированно частями и пройтись по нему за 5 неполных итераций. Т.е. каждый поток будет ответственнен за 4-5 отдельный областей поочередно. 
Вообще система хорошая: каждый поток имеет 4-5 отдельных заданий, спрайт, загруженный в шаред мемори и можно по частям проходиться по любого размера 
спрайтам. Это, конечно, все прикольно, но тогда, получается, у нас по одному блоку на кернел? А, нет, стоп. По спрайту на кернел. А в кернеле кол-во блоков 
будет равно кол-ву сайзингов. Т.е. на нашем случае будет 40 кернелов, по 22 блока в каждом, и у каждого блока по 1024 потока. Кол-во потоков - 901_120. 
Каждый поток будет выполнять в среднем 4.5 * 120 * 40 = 21_600 операций с, в основном, шаред памятью. Пока что с моим текущим пониманием того, как это 
работает, мне это кажется отличным вариантом. И кстати, регистр и сайзинги в любом случае должны с запасом влезть в константную память.
*/

//Наполняем карты пустот. Эти карты должны быть по одной на сайзинг и размером с дату.
__global__ void populateVoidMaps(short sizingsCount, short spritesCount, char* sizings, char* registry, char* data, char* voidMaps)
{
	printf("Hello from videocard! sizingsCount = %d. spritesCount = %d\n", sizingsCount, spritesCount);


}

int main()
{
	string path = "P:\\U\\Some2DGame\\Cuda\\info\\data.bytes";
	tuple<char*, int> blobTuple = file_reader::readFile(path);
	char* blob = get<0>(blobTuple);
	int blobLength = get<1>(blobTuple);

	int metaLength;
	memcpy(&metaLength, blob, 4);
	int combinedDataOffset = metaLength + 4;

	short spritesCount;
	memcpy(&spritesCount, blob + combinedDataOffset + 2, 2);
	short sizingsBlobLength;
	memcpy(&sizingsBlobLength, blob + combinedDataOffset + 4, 2);

	short sizingsCount = sizingsBlobLength / 4;

	int registryStructureLength = 8;

	char* sizingsBlob = blob + combinedDataOffset + 6;
	char* registryBlob = sizingsBlob + sizingsBlobLength;
	int registryBlobLength = spritesCount * registryStructureLength;
	char* dataBlob = registryBlob + registryBlobLength;
	int dataBlobLength = blobLength - registryBlobLength - sizingsBlobLength - combinedDataOffset - 6;

	char* deviceSizingsPtr;
	cudaMalloc((void**)&deviceSizingsPtr, sizingsBlobLength);
	char* deviceRegistryPtr;
	cudaMalloc((void**)&deviceRegistryPtr, registryBlobLength);
	char* deviceDataPtr;
	cudaMalloc((void**)&deviceDataPtr, dataBlobLength);

	cudaMemcpy(deviceSizingsPtr, sizingsBlob, sizingsBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceRegistryPtr, registryBlob, sizingsBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDataPtr, dataBlob, sizingsBlobLength, cudaMemcpyHostToDevice);

	dim3 block(128);
	dim3 grid(dataBlobLength / block.x);

	int voidMapsCount = 0;
	for (size_t i = 0; i < sizingsCount; i++)
	{
		short sizingWidth = bit_converter::GetShort(sizingsBlob, i * 2);
		short sizingHeight = bit_converter::GetShort(sizingsBlob, sizingsCount * 2 + i * 2);
		for (size_t j = 0; j < spritesCount; j++)
		{
			short spriteWidth = bit_converter::GetShort(registryBlob, spritesCount * 4 + i * 2);
			short spriteHeight = bit_converter::GetShort(registryBlob, spritesCount * 6 + i * 2);

			voidMapsCount += (spriteWidth - sizingWidth) * (spriteHeight * sizingHeight);
		}
	}
	char* voidMaps;
	cudaMalloc((void**)voidMaps, voidMapsCount / 8 + 1); //для инфы о пустоте достаточно 1 бита, поэтому делим кол-во в байтах на 8 с округлением в большую сторону
	populateVoidMaps << <grid, block >> > (sizingsCount, spritesCount, deviceSizingsPtr, deviceRegistryPtr, deviceDataPtr, voidMaps);

	cudaDeviceSynchronize();

	cudaFree(deviceSizingsPtr);
	cudaFree(deviceRegistryPtr);
	cudaFree(deviceDataPtr);
	free(blob);

    return 0;
}
