#include "numpy_reader.h"
#include <fstream>
#include "file_reader.h"
#include <string>
#include "string_extensions.h"
#include <vector>

std::tuple<char*, int, std::vector<int>> numpy_reader::readInt8(std::string path)
{
	std::tuple<char*, int> rawFile = file_reader::readFile(path);
	char* arr = std::get<0>(rawFile);// ._Myfirst._Val;
	if ((unsigned char)arr[0] != 147)
	{
		//printf("0 byte: %d", (unsigned char)arr[0]);
		throw "This in no numpy array. I don't know what this is.";
	}

	int len = std::get<1>(rawFile);// ._Get_rest()._Myfirst._Val;
	short lengthOfHeader;
	std::memcpy(&lengthOfHeader, arr + 8, 2);

	arr[10 + lengthOfHeader - 1] = 0;
	std::string header(arr + 10);
	std::string shapeStr = "shape': (";
	size_t shapePos = header.find(shapeStr);
	if (shapePos == std::string::npos)
		throw "This looks like numpy array, but shape not found!";

	std::vector<int> shapeVector;
	int currentPos = shapePos + shapeStr.size();
	int lastPos = currentPos;
	while (true)
	{
		char currentChar = header[lastPos];
		if (currentChar == ',' || currentChar == ')')
		{
			char* shapeEntryStr = string_extensions::makeSubset(header.c_str(), currentPos, lastPos);
			int shapeEntry = std::stoi(shapeEntryStr);
			shapeVector.push_back(shapeEntry);

			currentPos = lastPos + 1;
		}

		if (currentChar == ')')
			break;

		lastPos++;
	}

	char* result = arr + 10 + lengthOfHeader;
	return { result, len - 10 - lengthOfHeader, shapeVector };
}

std::tuple<int*, int, std::vector<int>> numpy_reader::readInt32(std::string path)
{
	std::tuple<char*, int, std::vector<int>> charResult = readInt8(path);
	char* arr = std::get<0>(charResult);// ._Myfirst._Val;
	int len = std::get<1>(charResult);// ._Get_rest()._Myfirst._Val;
	std::vector<int> shape = std::get<2>(charResult);// ._Get_rest()._Get_rest()._Myfirst._Val;
	int* result = (int*)(arr);
	return { result, len / 4, shape };
}
