#include "format_packer.h"
#include <tuple>
#include <stdio.h>
#include "bit_converter.h"

char getBitsCount(int number)
{
	if (number < 2)
		return 1;
	if (number < 4)
		return 2;
	if (number < 8)
		return 3;
	if (number < 16)
		return 4;
	if (number < 32)
		return 5;
	if (number < 64)
		return 6;
	if (number < 128)
		return 7;
	if (number < 256)
		return 8;
	if (number < 512)
		return 9;
	if (number < 1024)
		return 10;
	if (number < 2048)
		return 11;
	if (number < 4096)
		return 12;
	if (number < 8192)
		return 13;
	if (number < 16384)
		return 14;
	if (number < 32768)
		return 15;
	if (number < 65536)
		return 16;
	if (number < 131072)
		return 17;
	if (number < 262144)
		return 18;
	if (number < 524288)
		return 19;
	if (number < 1048576)
		return 20;
	if (number < 2097152)
		return 21;
	if (number < 4194304)
		return 22;
	if (number < 8388608)
		return 23;
	if (number < 16777216)
		return 24;
	if (number < 33554432)
		return 25;
	if (number < 67108864)
		return 26;
	if (number < 134217728)
		return 27;
	if (number < 268435456)
		return 28;
	if (number < 536870912)
		return 29;
	if (number < 1073741824)
		return 30;
	//if (number < 2_147_483_648) //int range exceeded
	return 31;
	//return 32;
}
int getMaxInteger(int size)
{
	/*if (size == 1)
		return 1;
	else if (size == 2)
		return 3;
	else if (size == 3)
		return 7;
	else if (size == 4)
		return 15;
	else if (size == 5)
		return 31;
	else if (size == 6)
		return 63;
	else if (size == 7)
		return 127;
	else if (size == 8)
		return 255;
	else if (size == 9)
		return 511;
	else if (size == 10)
		return 1023;
	else if (size == 11)
		return 2047;
	else if (size == 12)
		return 4095;
	else if (size == 13)
		return 8191;
	else if (size == 14)
		return 16383;
	else if (size == 15)
		return 32767;
	else if (size == 16)
		return 65535;
	else if (size == 17)
		return 131071;
	else if (size == 18)
		return 262143;
	else if (size == 19)
		return 524287;
	else if (size == 20)
		return 1048575;
	else if (size == 21)
		return 2097151;
	else if (size == 22)
		return 4194303;
	else if (size == 23)
		return 8388607;
	else if (size == 24)
		return 16777215;
	else if (size == 25)
		return 33554431;
	else if (size == 26)
		return 67108863;
	else if (size == 27)
		return 134217727;
	else if (size == 28)
		return 268435455;
	else if (size == 29)
		return 536870911;
	else if (size == 30)
		return 1073741823;
	else if (size == 31)
		return 2147483;*/
	if (size == 1)
		return 0b00000000000000000000000000000001;
	else if (size == 2)
		return 0b00000000000000000000000000000011;
	else if (size == 3)
		return 0b00000000000000000000000000000111;
	else if (size == 4)
		return 0b00000000000000000000000000001111;
	else if (size == 5)
		return 0b00000000000000000000000000011111;
	else if (size == 6)
		return 0b00000000000000000000000000111111;
	else if (size == 7)
		return 0b00000000000000000000000001111111;
	else if (size == 8)
		return 0b00000000000000000000000011111111;
	else if (size == 9)
		return 0b00000000000000000000000111111111;
	else if (size == 10)
		return 0b00000000000000000000001111111111;
	else if (size == 11)
		return 0b00000000000000000000011111111111;
	else if (size == 12)
		return 0b00000000000000000000111111111111;
	else if (size == 13)
		return 0b00000000000000000001111111111111;
	else if (size == 14)
		return 0b00000000000000000011111111111111;
	else if (size == 15)
		return 0b00000000000000000111111111111111;
	else if (size == 16)
		return 0b00000000000000001111111111111111;
	else if (size == 17)
		return 0b00000000000000011111111111111111;
	else if (size == 18)
		return 0b00000000000000111111111111111111;
	else if (size == 19)
		return 0b00000000000001111111111111111111;
	else if (size == 20)
		return 0b00000000000011111111111111111111;
	else if (size == 21)
		return 0b00000000000111111111111111111111;
	else if (size == 22)
		return 0b00000000001111111111111111111111;
	else if (size == 23)
		return 0b00000000011111111111111111111111;
	else if (size == 24)
		return 0b00000000111111111111111111111111;
	else if (size == 25)
		return 0b00000001111111111111111111111111;
	else if (size == 26)
		return 0b00000011111111111111111111111111;
	else if (size == 27)
		return 0b00000111111111111111111111111111;
	else if (size == 28)
		return 0b00001111111111111111111111111111;
	else if (size == 29)
		return 0b00011111111111111111111111111111;
	else if (size == 30)
		return 0b00111111111111111111111111111111;
	else if (size == 31)
		return 0b01111111111111111111111111111111;
	return 0b11111111111111111111111111111111;
}

/*
	Ок, нам надо записывать биты. Допустим нам пришло bitIndex 41, value 10, valueFrameLength 27

	Буффер у нас выглядит примерно так
	| байт | байт | байт | байт | байт | байт | и т.д.
	0      8      16     24     32     40

	Максимум, что нас попросят записать - это полный инт, следовательно вообще-то нам надо квантовать по интам.
	Мы берем 1й инт буффер - начинающийся с 32го бита и видим, что мы можем начать писать в него с 9го.

	var startBitIndex = bitIndex % 32;
	var availableSpace = 32 - startBitIndex;

	Теперь нам надо записать данные value длиной в availableSpace.
	У нас есть начало инта, длиной в startBitIndex, которое надо сохранить...

	11100110 1------- -------- --------

	...и окончание инта, начало valueFrameLength, которое надо записать

	00000000 00000000 00000000 00001010
	длиной в 27, т.е.
	-----000 00000000 00000000 00001010

	А доступна у нас вот эта область
	-------- -1111111 11111111 11111111

	Тогда,

	if (availableSpace < valueFrameLength)
	{
		var firstChunk = (value >> (valueFrameLength - availableSpace)) & getMaxInteger(availableSpace);
		var secondChunk = value & getMaxInteger(valueFrameLength - availableSpace)
	}

	valueFrameLength - availableSpace это у нас 27 - 23 = 4. Таким образов firstChunk у нас будет
	-------- -0000000 00000000 00000000
	А secondChunk
	-------- -------- -------- ----1010

	Дальше нам надо записать эти два чанка в буффер. В случае с первым чанком у нас будет информация в начале буффера, которую требуется сохранить.
	Собственно в конце буффера и начале value у нас нули, поэтому сработает обчный |.

	А что в случаем когда у нас хватает места?

	Ок, по-видимому сдвигать надо в другую сторону.


	Ок, я думал, что у нас вот так
	00000001 00010100 00000000 00000000
	00000000 -------- -------- --------

	А оказалоь, что на самом деле вот эдак
	00000000 00000000 00010100 00000001
	-------- -------- -------- 00000000 //резерв

	Если у нас так, то, чтобы записать 1й байт после резерва, мне надо сдвинуть его на startBitIndex влево

	theOneChunk = value << startBitIndex;

	А если бы мне надо было записать все 4 байта, но понадобилось бы 2 куска. 
	1) value & maxInt(availableSpace) << startBitIndex
	2) value >> availableSpace
*/

int bitwiseWrite(int* buffer, int bitIndex, int value, char valueFrameLength)
{
	int bufferIndex = bitIndex / 32;
	int startBitIndex = bitIndex % 32;
	int availableSpace = 32 - startBitIndex;
	unsigned int uValue;
	memcpy(&uValue, &value, 4);
	if (bitIndex <= 40)
		printf("bitIndex = %d\n", bitIndex);
	if (bitIndex <= 40)
		printf("value = %d\n", value);
	if (bitIndex <= 40)
		printf("uValue = %u\n", uValue);
	if (bitIndex <= 40)
		printf("valueFrameLength = %d\n", valueFrameLength);
	if (availableSpace < valueFrameLength)
	{
		int firstChunk = (uValue & getMaxInteger(availableSpace)) << startBitIndex;
		int secondChunk = uValue >> availableSpace;
		if (bitIndex <= 40)
			printf("firstChunk = %d\n", firstChunk);
		if (bitIndex <= 40)
			printf("secondChunk = %d\n", secondChunk);
		if (bitIndex <= 40)
			printf("buffer[bufferIndex] before = %d\n", buffer[bufferIndex]);
		if (bitIndex <= 40)
			printf("buffer[bufferIndex + 1] before = %d\n", buffer[bufferIndex + 1]);

		buffer[bufferIndex] |= firstChunk;
		buffer[bufferIndex + 1] = secondChunk;
		if (bitIndex <= 40)
			printf("buffer[bufferIndex] after = %d\n", buffer[bufferIndex]);
		if (bitIndex <= 40)
			printf("buffer[bufferIndex + 1] after = %d\n", buffer[bufferIndex + 1]);
	}
	else
	{
		int theOneChunk = uValue << startBitIndex;
		if (bitIndex <= 40)
			printf("theOneChunk = %d\n", theOneChunk);
		if (bitIndex <= 40)
			printf("buffer[bufferIndex] before = %d\n", buffer[bufferIndex]);
		buffer[bufferIndex] |= theOneChunk;
		if (bitIndex <= 40)
			printf("buffer[bufferIndex] after = %d\n", buffer[bufferIndex]);
	}
	return bitIndex + valueFrameLength;
}

/*
Ок, у нас есть по идее все, что нужно, чтобы упаковать все в формат.

Для начала у нас есть длина атласа - кол-во кусков-победителей, так мы можем определить максимальные длины их измерений.

Ок, в блобплеере на той стороне у нас оффсеты упакованы очень оптимизировано - исходя из необходимого кол-ва спрайтов на секвенцию.
Но это не подходит к случаю, когда мы не знаем заранее какие секвенции и сколько их одновременно нам понадобятся. Какие-то
спрайты могут быть частью секвенции, а какие-то могут быть просто одинарными картинками. Такая упаковка подходит только для
случая с заранее известной секвенцией, а на сервере мы не можем об этом заранее знать. Поэтому сервер должен отдавать данные в
более-менее универсализированном виде, чтобы потом клиент мог из этого легко слепить то, что ему будет нужно.

Отличие данных форматов по сути только в оптимизации показывания чанков. Анимация состоит из определенного набора спрайтов,
которых нам гарантированно хватит, чтобы показать любой кадр анимации.

Чтобы потом было легко оптимизировать под конкретные нужды нам нужно знать кол-во чанков на спрайт, т.к. именно спрайт - неделимая
базовая единица отображения графики в игре (не попиксельно же!).

После атласа мне надо хранить инфу о смещениях. Там у нас структура из номера чанка, х и у. Они будут идти подряд, и порядок будет
важен с случаях когда мы будем отключать альфу. Чтобы понять когда начинаются и заканчиваются спрайты нам надо хранить их оффсеты.
Итак, сначала оффсеты, потом сами данные. Данных может быть много и оффсеты могут быть большими. Но я не знаю, могут ли они быть
больше шорта. Наверное могут. Этих оффсетов может быть настолько много, что лучше хранить отдельно длину оффсета.

2.2.2020
Ок, был неправ. Индексы должны быть не такими. Во-первых у кусков атласа индексы обозначают индекс спрайта, поэтому переименовал в
atlasChunkSpriteIndexLength. При этом нельзя просто взять bitsCount(spritesCount), т.к. из может быть очень много, а куски атласа
могут браться только из пары первых спрайтов и мы будем waste space. Второе - индексы смещений - они указывают уже на индекс куска
в атласе. Они используются все, поэтому длина этого индекса будет bistCount(atlasLength).
*/

std::tuple<char*, int> format_packer::pack(int atlasLength, char* atlasBuffer, unsigned int* offsetsBuffer, int spritesCount, short* spriteWidths, short* spriteHeights, char* prefixBuffer, int prefixLength)
{
	unsigned char atlasChunkSpriteIndexLength = 0;
	unsigned char atlasXLength = 0;
	unsigned char atlasYLength = 0;
	unsigned char atlasWidthLength = 0;
	unsigned char atlasHeightLength = 0;

	unsigned char atlasLengthLength = getBitsCount(atlasLength);
	unsigned char chunkOffsetXLength = 0;
	unsigned char chunkOffsetYLength = 0;

	printf("pack 1\n");

	int atlasStructureSize = sizeof(int) + sizeof(short) * 4;
	for (size_t i = 0; i < atlasLength; i++)
	{
		int spriteIndex = ((int*)(atlasBuffer + i * atlasStructureSize))[0];
		int chunkX = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int)))[0];
		int chunkY = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int) + sizeof(short)))[0];
		int chunkWidth = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int) + sizeof(short) * 2))[0];
		int chunkHeight = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int) + sizeof(short) * 3))[0];

		char chunkSpriteIndexLength = getBitsCount(spriteIndex);
		if (chunkSpriteIndexLength > atlasChunkSpriteIndexLength)
			atlasChunkSpriteIndexLength = chunkSpriteIndexLength;
		char chunkXLength = getBitsCount(chunkX);
		if (chunkXLength > atlasXLength)
			atlasXLength = chunkXLength;
		char chunkYLength = getBitsCount(chunkY);
		if (chunkYLength > atlasYLength)
			atlasYLength = chunkYLength;
		char chunkWidthLength = getBitsCount(chunkWidth);
		if (chunkWidthLength > atlasWidthLength)
			atlasWidthLength = chunkWidthLength;
		char chunkHeightLength = getBitsCount(chunkHeight);
		if (chunkHeightLength > atlasHeightLength)
			atlasHeightLength = chunkHeightLength;
	}
	printf("pack 2\n");

	int maxChunksInSpriteCount = 0;
	int overallChunksCount = 0;
	int offsetSoFar = 0;
	for (size_t i = 0; i < spritesCount; i++)
	{
		int chunksInSpriteCount = 0;
		for (size_t x = 0; x < spriteWidths[i]; x++)
		{
			for (size_t y = 0; y < spriteHeights[i]; y++)
			{
				int pixelIndex = offsetSoFar + x * spriteHeights[i] + y;
				if (offsetsBuffer[pixelIndex] != 0)
				{
					char currentOffsetXLength = getBitsCount(x);
					char currentOffsetYLength = getBitsCount(y);
					if (currentOffsetXLength > chunkOffsetXLength)
						chunkOffsetXLength = currentOffsetXLength;
					if (currentOffsetYLength > chunkOffsetYLength)
						chunkOffsetYLength = currentOffsetYLength;
					chunksInSpriteCount++;
				}
			}
		}

		if (maxChunksInSpriteCount < chunksInSpriteCount)
			maxChunksInSpriteCount = chunksInSpriteCount;

		overallChunksCount += chunksInSpriteCount;

		offsetSoFar += spriteWidths[i] * spriteHeights[i];
	}

	int maxChunksInSpriteCountLength = getBitsCount(maxChunksInSpriteCount);

	unsigned int bufferLengthInBits = 8 //1 байт зарезервирован.
		+ prefixLength * 8 //Мета-инфа. Она у нас в байтах, поэтому x8
		+ 32 //atlasLength. Всякое бывает, может и 4 байта
		+ 32 //spritesCount. Спрайтов может и будет больше 65к, поэтому чтобы не мелочиться берем 4 байта
		+ 16 //maxChunksInSpriteCountLength. Обозначает длину значения кол-ва кусков атласа в спрайте. Очень нужно

		+ 16 //atlasChunkSpriteIndexLength - длина, требующаяся индексу спрайта куска атласа
		+ 16 //atlasXLength. Разрешение не может быть больше 65Кх65К ни в каком обозримом будущем
		+ 16 //atlasYLength
		+ 16 //atlasWidthLength
		+ 16 //atlasHeightLength

		+ 32 //atlasLengthLength //Длина кол-ва кусков атласа - как раз хватит для описания идекса куска в атласе для отдельных чанков
		+ 16 //chunkOffsetXLength //Смещения указанного индексом куска в отдельных спрайтах
		+ 16 //chunkOffsetYLength

		+ atlasChunkSpriteIndexLength * atlasLength //Дальше идут записи атласа.
		+ atlasXLength * atlasLength
		+ atlasYLength * atlasLength
		+ atlasWidthLength * atlasLength
		+ atlasHeightLength * atlasLength

		+ spritesCount * maxChunksInSpriteCountLength //По оффсету на каждый спрайт - на начало записей кусков для каждого

		+ atlasLengthLength * overallChunksCount //Непосредственно записи смещений кусков атласов - (atlas index, chunk (x, y))
		+ chunkOffsetXLength * overallChunksCount
		+ chunkOffsetYLength * overallChunksCount;

	unsigned int bufferLength = bufferLengthInBits / 8;
	if (bufferLengthInBits % 8 != 0)
		bufferLength++;

	int* buffer = (int*)malloc(bufferLength);
	if (buffer != NULL)
	{
		printf("!= NULL");
		memset(buffer, 0, bufferLength);
	}


	int metaLength = bit_converter::GetInt(prefixBuffer, 0);
	printf("metaLength = %d\n", metaLength);
	metaLength = bit_converter::GetInt((char*)buffer, 1);
	printf("metaLength2 = %d\n", metaLength);
	int bitIndex = 8;
	for (size_t i = 0; i < prefixLength; i++)
	{
		if (bitIndex <= 40)
		{
			printf("bitIndex = %d\n", bitIndex);
			printf("1st byte source: %d\n", (int)prefixBuffer[0]);
			printf("1st byte before: %d\n", (int)((char*)buffer)[1]);
			printf("2d byte source: %d\n", (int)prefixBuffer[1]);
			printf("2d byte before: %d\n", (int)((char*)buffer)[2]);
			printf("3d byte source: %d\n", (int)prefixBuffer[2]);
			printf("3d byte before: %d\n", (int)((char*)buffer)[3]);
			printf("4th byte source: %d\n", (int)prefixBuffer[3]);
			printf("4th byte before: %d\n", (int)((char*)buffer)[4]);
		}
		bitIndex = bitwiseWrite(buffer, bitIndex, (unsigned char)prefixBuffer[i], 8);
		/*if (bitIndex == 16)
			printf("1st byte after: %u\n", (unsigned int)buffer[1]);*/

		if (bitIndex <= 48)
		{
			printf("1st byte after: %d\n", (int)((char*)buffer)[1]);
			printf("2d byte after: %d\n", (int)((char*)buffer)[2]);
			printf("3d byte after: %d\n", (int)((char*)buffer)[3]);
			printf("4th byte after: %d\n", (int)((char*)buffer)[4]);
		}
	}
	metaLength = bit_converter::GetInt((char*)buffer, 1);
	printf("metaLength3 = %d\n", metaLength);

	printf("atlasLength1 = %d\n", atlasLength);
	printf("bitIndex = %d, byteIndex = %d\n", bitIndex, bitIndex / 8);
	bitIndex = bitwiseWrite(buffer, bitIndex, atlasLength, 32);
	printf("atlasLength2 = %d, byteIndex = %d\n", bit_converter::GetInt((char*)buffer, (bitIndex / 8) - 4), (bitIndex / 8) - 4);
	bitIndex = bitwiseWrite(buffer, bitIndex, spritesCount, 32);
	bitIndex = bitwiseWrite(buffer, bitIndex, maxChunksInSpriteCountLength, 16);

	bitIndex = bitwiseWrite(buffer, bitIndex, atlasChunkSpriteIndexLength, 16);
	bitIndex = bitwiseWrite(buffer, bitIndex, atlasXLength, 16);
	bitIndex = bitwiseWrite(buffer, bitIndex, atlasYLength, 16);
	bitIndex = bitwiseWrite(buffer, bitIndex, atlasWidthLength, 16);
	bitIndex = bitwiseWrite(buffer, bitIndex, atlasHeightLength, 16);

	bitIndex = bitwiseWrite(buffer, bitIndex, atlasLengthLength, 32);
	bitIndex = bitwiseWrite(buffer, bitIndex, chunkOffsetXLength, 16);
	bitIndex = bitwiseWrite(buffer, bitIndex, chunkOffsetYLength, 16);

	//Записываем атлас

	for (size_t i = 0; i < atlasLength; i++)
	{
		int spriteIndex = ((int*)(atlasBuffer + i * atlasStructureSize))[0];
		short chunkX = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int)))[0];
		short chunkY = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int) + sizeof(short)))[0];
		short chunkWidth = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int) + sizeof(short) * 2))[0];
		short chunkHeight = ((short*)(atlasBuffer + i * atlasStructureSize + sizeof(int) + sizeof(short) * 3))[0];

		bitIndex = bitwiseWrite(buffer, bitIndex, spriteIndex, atlasChunkSpriteIndexLength);
		bitIndex = bitwiseWrite(buffer, bitIndex, chunkX, atlasXLength);
		bitIndex = bitwiseWrite(buffer, bitIndex, chunkY, atlasYLength);
		bitIndex = bitwiseWrite(buffer, bitIndex, chunkWidth, atlasWidthLength);
		bitIndex = bitwiseWrite(buffer, bitIndex, chunkHeight, atlasHeightLength);
	}

	//Записываем чанки

	offsetSoFar = 0;
	for (size_t i = 0; i < spritesCount; i++)
	{
		int chunksInSpriteCount = 0;
		for (size_t x = 0; x < spriteWidths[i]; x++)
		{
			for (size_t y = 0; y < spriteHeights[i]; y++)
			{
				int pixelIndex = offsetSoFar + x * spriteHeights[i] + y;
				if (offsetsBuffer[pixelIndex] != 0)
				{
					chunksInSpriteCount++;
				}
			}
		}

		bitIndex = bitwiseWrite(buffer, bitIndex, chunksInSpriteCount, maxChunksInSpriteCountLength);
		offsetSoFar += spriteWidths[i] * spriteHeights[i];
	}
	printf("bufferLengthInBits = %d\n", bufferLengthInBits);
	printf("bitIndex before chunks writing: %d\n", bitIndex);
	printf("atlasLengthLength = %d, chunkOffsetXLength = %d, chunkOffsetYLength = %d\n", atlasLengthLength, chunkOffsetXLength, chunkOffsetYLength);

	overallChunksCount = 0;
	offsetSoFar = 0;
	for (size_t i = 0; i < spritesCount; i++)
	{
		for (size_t x = 0; x < spriteWidths[i]; x++)
		{
			for (size_t y = 0; y < spriteHeights[i]; y++)
			{
				int pixelIndex = offsetSoFar + x * spriteHeights[i] + y;
				if (offsetsBuffer[pixelIndex] != 0)
				{
					bitIndex = bitwiseWrite(buffer, bitIndex, offsetsBuffer[pixelIndex] - 1, atlasLengthLength);
					bitIndex = bitwiseWrite(buffer, bitIndex, x, chunkOffsetXLength);
					//printf("i = %zd, x = %zd, y = %zd, bitIndex = %d, chunkOffsetYLength = %d\n", i, x, y, bitIndex, chunkOffsetYLength);
					bitIndex = bitwiseWrite(buffer, bitIndex, y, chunkOffsetYLength);
					overallChunksCount++;
				}
			}
		}
		offsetSoFar += spriteWidths[i] * spriteHeights[i];
	}
	printf("overallChunksCount (2) = %d\n", overallChunksCount);
	printf("bitIndex after chunks writing: %d\n", bitIndex);

	return { (char*)buffer, bufferLength };
}