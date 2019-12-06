
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

Стоп, я что-то не додумал... Сравнивать-то мне надо будет со ВСЕМИ спрайтам. Так что память будет использоваться не только шаред, но и общая. Ну уместить всю
дату в шаред мемори не представляется возможным... Хотя, если распределить задачи по времени... Чтобы в кернеле было не 22 блока а 40х22 = 880 блоков! В
каждом по 1024 потока. После завершения всех - тупо складываем результат по сайзингам, будет 22*120*40 результата на кернел как и надо. Результатов будет
много - 22*120*40*40, более 4 миллионов. Сложить их можно будет также параллельно - завершающим складывающим кернелом. На выходе он нам сделает наши 105_600
результатов - по 4800 (120х40) результатов на спрайт для каждого из 22х сайзингов. Со всех 40 кернелов получится обратно 4 миллиона результатов, из которых
мне самой элементарной сортировкой надо будет отобрать наибольший результат. Самой элементарной - это значит запускаем блоки по 1024 потока, агрессивными
оптимизациями в 10 проходов получаем 1 наибольший, пишем в буфер, и проводим с ним ту же операцию, пока не останется 1 значение. Шикарно.

А войдмапы пускай клиенты делают - это не особо их затруднит, я думаю.

P.S. Тут только проблемы с адресацией опять же возникнут, но я уже придумал как их решить. Составляем заранее (опять же - клиентом) мапу... Хотя нет, даже
этого не нужно. Каждый блок знает заранее с какого спрайта у него референс, а с какого претендер. Так что он просто сразу знает и оба оффсета. Только вот еще
одна проблема потенциальная - т.к. мы занимаем память, скорее всего, по максимуму, на одном мультипроцессоре будет у нас жить тоько 1 блок. А работать этот
блок будет с глобал мемори. Это значит большие задержки и простой мультипроцессоров. Решение здесь - ограничить сильнее использование шаред мемори - пусть
лучше будет больше итераций. Т.е. вообще-то кол-ву памяти будет достаточно быть равным кол-ву потоков - 1024 потокам не нужно больше инфы в шаред мемори, чем
1024 + полщадь сайзинга (8х8=64 в нашем случае). Если на каждый пиксель надо будет иметь 5 байтов (4 байта цвета + 1 байт пустоты), это всего лишь около 5.3кб
памяти на блок в одно и то же время. Это значит на одном мультипроцессоре смогут сожительствовать 18 блоков. Думаю, может быть норм, особенно учитывя еще, что
в 1 процессоре одновременно могут проходить только 4 варпа, но надо смотреть. Вообще круто, что на моей карточке 96Кб общей памяти - наибольшее кол-во из
текущих. Посмотрим, что принесет 8.0 compute capability... В общем, если будет не хватать памяти, буду уменьшать размер блока и соответсвенно размер требуемой
шаред мемори на блок, таким образом увеличивая кол-во блоков на процессор.


5.12.2019
Все-таки решил заполнять карты пустот самому. Ибо для больших наборов спрайтов (а скорее всего наборы будут большими) эти карты пустот будут большого размера.
Для 100 мегапикселей например, карта пустот с 22 сайзингами будет дополнительными 300 мегабайтами информации. Есть мнение (мое), что это быстрее обсчитается
на видеокарте, чем загрузится в память... Хотя все же нет, 300 мб не так много. Это во-первых. А во-вторых вычислительные мощности - дефицитный ресурс, узкое
место. Даже если все это и обсчитается быстрее, чем прочитается с диска и загрузится, шине и диску все равно делать нефига, а за то время пока они грузят,
карта обсчитает больше действителньо нужной инфы.

P.S. Я ошибся - войдмапы занимают меньше места: не в 3 раза больше для 22 сайзингов, а всего где-то 3/4 размера. Забыл, что в каждом пикселе еще по 4 байта.


6.12.2019
Ок, я что-то не подумал. Кернелы не могут выполняться по-настоящему параллельно. Поэтому надо делать все одним кернелом. Какие следствия из этого? Да
собственно не такие уж большие, раз я решил разбивать задачи по блокам не равномерно. Просто теперь blockIdx.x будет содержать трехмерным.
*/

//Наполняем карты пустот. Эти карты должны быть по одной на сайзинг и размером с дату.
__global__ void mainKernel(int sizingsCount, short* sizingWidths, short* sizingHeights, int spritesCount, int* offsets, short* widths, short* heights, unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a)
{
	int ourSpriteIndex = blockIdx.x;
	int candidateSpriteIndex = blockIdx.y;
	int sizingIndex = blockIdx.z;

	int ourOffset = offsets[ourSpriteIndex];
	int ourWidth = widths[ourSpriteIndex];
	int ourHeight = heights[ourSpriteIndex];

	int candidateSpriteOffset = offsets[candidateSpriteIndex];
	int candidateSpriteWidth = widths[candidateSpriteIndex];
	int candidateSpriteHeight = heights[candidateSpriteIndex];

	int sizingWidth = sizingWidths[sizingIndex];
	int sizingHeight = sizingHeights[sizingIndex];

	if (threadIdx.x == 0)
		printf("Hello from block! My sprite is #%d (width %d, height %d) and I work with sprite %d (width %d, height %d) and sizing %d (width %d, height %d) \n", ourSpriteIndex, ourWidth, ourHeight, candidateSpriteIndex, candidateSpriteWidth, candidateSpriteHeight, sizingIndex, sizingWidth, sizingHeight);
}

__global__ void testKernel(int sizingsCount, short* sizingWidths, short* sizingHeights, int spritesCount, int* offsets, short* widths, short* heights, unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* a)
{
	if (threadIdx.x == 0)
	{
		printf("%d\n", threadIdx.x);
		return;
	}
	int ourSpriteIndex = blockIdx.x;
	int candidateSpriteIndex = blockIdx.y;
	int sizingIndex = blockIdx.z;

	int ourOffset = offsets[ourSpriteIndex];
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
	//int dataBlobLength = blobLength - registryBlobLength - sizingsBlobLength - combinedDataOffset - 6;
	//char* dataBlobs = new char[spritesCount];

	int dataBlobLineLength = 0;
	for (size_t i = 0; i < spritesCount; i++)
	{
		int width = bit_converter::GetShort(registryBlob, spritesCount * 4 + i * 2);
		int height = bit_converter::GetShort(registryBlob, spritesCount * 6 + i * 2);
		dataBlobLineLength += width * height;
	}
	int dataBlobLength = dataBlobLineLength * 4;
	char* voidsBlob = dataBlob + dataBlobLength;
	//int voidsBlobLength = (dataBlobLength / 32 * sizingsCount) / 8 + 1;
	int voidsBlobLength = blobLength - dataBlobLength - registryBlobLength - sizingsBlobLength - combinedDataOffset - 6;


	char* deviceSizingsPtr;
	cudaMalloc((void**)&deviceSizingsPtr, sizingsBlobLength);
	char* deviceRegistryPtr;
	cudaMalloc((void**)&deviceRegistryPtr, registryBlobLength);
	char* deviceDataPtr;
	cudaMalloc((void**)&deviceDataPtr, dataBlobLength);
	char* deviceVoidsPtr;
	cudaMalloc((void**)&deviceVoidsPtr, voidsBlobLength);

	cudaMemcpy(deviceSizingsPtr, sizingsBlob, sizingsBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceRegistryPtr, registryBlob, registryBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDataPtr, dataBlob, dataBlobLength, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceVoidsPtr, voidsBlob, voidsBlobLength, cudaMemcpyHostToDevice);


	int* offsets = (int*)deviceRegistryPtr;
	int offsetsLength = spritesCount * 4;
	short* widths = (short*)(deviceRegistryPtr + offsetsLength);
	int widthsLength = spritesCount * 2;
	short* heights = (short*)(deviceRegistryPtr + offsetsLength + widthsLength);
	int heightsLength = widthsLength;

	short* sizingWidths = (short*)deviceSizingsPtr;
	int sizingWidthsLength = sizingsCount * 2;
	short* sizingHeights = (short*)(deviceSizingsPtr + sizingWidthsLength);
	int sizingHeightsLength = sizingWidthsLength;

	unsigned char* r = (unsigned char*)deviceDataPtr;
	unsigned char* g = (unsigned char*)(deviceDataPtr + dataBlobLineLength);
	unsigned char* b = (unsigned char*)(deviceDataPtr + dataBlobLineLength * 2);
	unsigned char* a = (unsigned char*)(deviceDataPtr + dataBlobLineLength * 3);

	dim3 block(1024);
	dim3 grid(spritesCount, spritesCount, sizingsCount); //Сайзингов будет меньше, чем спрайтов, так что сайзинги записываем в z
	mainKernel << <grid, block >> > (sizingsCount, sizingWidths, sizingHeights, spritesCount, offsets, widths, heights, r, g, b, a);

	cudaDeviceSynchronize();

	cudaFree(deviceSizingsPtr);
	cudaFree(deviceRegistryPtr);
	cudaFree(deviceDataPtr);
	cudaFree(deviceVoidsPtr);
	free(blob);

	return 0;
}
