
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include "file_reader.h"

using namespace std;

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


	cudaFree(deviceSizingsPtr);
	cudaFree(deviceRegistryPtr);
	cudaFree(deviceDataPtr);
	free(blob);

    return 0;
}
