#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Dirent.h"
#include "file_reader.h"
#include "bit_converter.h"

using namespace std;
namespace fs = experimental::filesystem;

int main()
{
	std::string path = "P:\\U\\Some2DGame\\Cuda\\info\\data.bytes";
	std::tuple<char*, int> blobTuple = file_reader::readFile(path);
	char* blob = get<0>(blobTuple);

	int blobLength = get<1>(blobTuple);

	int metaLength;
	std::memcpy(&metaLength, blob, 4);
	int combinedDataOffset = metaLength + 4;

	short spritesCount;
	std::memcpy(&spritesCount, blob + combinedDataOffset + 2, 2);
	short sizingsBlobLength;
	std::memcpy(&sizingsBlobLength, blob + combinedDataOffset + 4, 2);

	short sizingsCount = sizingsBlobLength / 4;

	int registryStructureLength = 8;

	char* sizingsBlob = blob + combinedDataOffset + 6;
	char* registryBlob = sizingsBlob + sizingsBlobLength;
	int registryBlobLength = spritesCount * registryStructureLength;
	char* dataBlob = registryBlob + registryBlobLength;
	int dataBlobLength = blobLength - registryBlobLength - sizingsBlobLength - combinedDataOffset - 6;


	/*printf("metaLength: %d\n", metaLength);

	printf("spritesCount: %d\n", spritesCount);
	printf("sizingsCount: %d\n", sizingsCount);

	printf("sizings block length: %d\n", sizingsBlobLength);
	printf("sizings[0]: %d\n", bit_converter::GetShort(sizingsBlob, 0));
	printf("sizings[1]: %d\n", bit_converter::GetShort(sizingsBlob, 2));
	printf("sizings[2]: %d\n", bit_converter::GetShort(sizingsBlob, 4));
	printf("sizings[3]: %d\n", bit_converter::GetShort(sizingsBlob, 6));
	printf("\n");

	printf("registry block length: %d\n", registryBlobLength);
	printf("registryBlob[0]: %d\n", bit_converter::GetInt(registryBlob, 0));
	printf("registryBlob[1]: %d\n", bit_converter::GetInt(registryBlob, 4));
	printf("registryBlob[2]: %d\n", bit_converter::GetInt(registryBlob, 8));
	printf("registryBlob[3]: %d\n", bit_converter::GetInt(registryBlob, 12));
	printf("\n");

	for (size_t i = 0; i < spritesCount; i++)
		printf("for registry %d: %d\n", i, bit_converter::GetInt(blob, metaLength + 4 + 6 + sizingsBlobLength + i * 4));
	for (size_t i = 0; i < spritesCount; i++)
		printf("for registry %d: %d\n", i, bit_converter::GetShort(blob, metaLength + 4 + 6 + sizingsBlobLength + spritesCount * 4 + i * 2));
	for (size_t i = 0; i < spritesCount; i++)
		printf("for registry %d: %d\n", i, bit_converter::GetShort(blob, metaLength + 4 + 6 + sizingsBlobLength + spritesCount * 6 + i * 2));

	printf("\n");
	printf("datablobpos: \n");
	printf("dataBlob[0]: %d\n", (int)bit_converter::GetChar(dataBlob, 0));
	printf("dataBlob[1]: %d\n", (int)bit_converter::GetChar(dataBlob, 1));
	printf("dataBlob[2]: %d\n", (int)bit_converter::GetChar(dataBlob, 2));
	printf("dataBlob[3]: %d\n", (int)bit_converter::GetChar(dataBlob, 3));
	printf("dataBlob[4]: %d\n", (int)bit_converter::GetChar(dataBlob, 4));
	printf("dataBlob[5]: %d\n", (int)bit_converter::GetChar(dataBlob, 5));
	printf("dataBlob[6]: %d\n", (int)bit_converter::GetChar(dataBlob, 6));
	printf("dataBlob[7]: %d\n", (int)bit_converter::GetChar(dataBlob, 7));
	printf("dataBlob[8]: %d\n", (int)bit_converter::GetChar(dataBlob, 8));
	printf("dataBlob[9]: %d\n", (int)bit_converter::GetChar(dataBlob, 9));
	printf("dataBlob[10]: %d\n", (int)bit_converter::GetChar(dataBlob, 10));
	printf("dataBlob[11]: %d\n", (int)bit_converter::GetChar(dataBlob, 11));
	printf("dataBlob[12]: %d\n", (int)bit_converter::GetChar(dataBlob, 12));
	printf("dataBlob[13]: %d\n", (int)bit_converter::GetChar(dataBlob, 13));
	printf("dataBlob[14]: %d\n", (int)bit_converter::GetChar(dataBlob, 14));
	printf("dataBlob[15]: %d\n", (int)bit_converter::GetChar(dataBlob, 15));
	printf("\n");*/
}