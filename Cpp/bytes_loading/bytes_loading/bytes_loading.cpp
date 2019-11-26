#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Dirent.h"
#include "numpy_reader.h"

using namespace std;
namespace fs = experimental::filesystem;

char* readFileBytes(const char* name);

int main()
{
	std::string path = "P:\\U\\Some2DGame\\Py\\info";
	DIR* directory = opendir(path.c_str());
	struct dirent* direntStruct;

	/*
	Ок, тут мне нужно сложить все данные в один блоб, и записать все в регистр, чтобы как-то в нем ориентироваться.
	Это по сути все, что мне нужно для перенесения старой логики в имплементацию на CUDA. Остальные буфферы - это уже
	служебные всякие ну для поддержки многочисленных смен контекста цпу<->гпу. Тут мы все сразу отправим и примем только
	уже конечный результат, для которого, конечно тоже понадобится буффер, но это уже ближе к делу.

	В плане производительности желательно хранить еще и структуру массивов, а не массив структур, так что надо будет еще 
	и немного переделывать этот входящий блоб.

	Короче когда я получаю char* я получаю трехмерный массив, примерно такой формы - 16, 24, 4. Мне надо поменять 
	размерность на 4, 16, 24.

	Вообще-то я туплю. Все это можно и нужно делать на стороне клиента. Чтобы на серверной стороне уже только послать все
	это на гпу.
	*/

	if (directory != NULL) {

		while (direntStruct = readdir(directory)) {
			string fullpath(path);
			fullpath.append("\\");
			fullpath.append(direntStruct->d_name);
			if (fullpath.find(".npy") != std::string::npos)
			{
				printf("File Name: %s\n", fullpath.c_str()); //If you are using <stdio.h>
				if (fullpath.find("sizings") != std::string::npos)
					std::tuple<int*, int, std::vector<int>> sizings = numpy_reader::readInt32(fullpath);
				else
					std::tuple<char*, int, std::vector<int>> sizings = numpy_reader::readInt8(fullpath);
			}
			//std::cout << direntStruct->d_name << std::endl; //If you are using <iostream>
		}
	}
	closedir(directory);

	/*char* fileName = nullptr;
	while (fileGetter.getNextFile(fileName))
		std::cout << fileName << '\n';*/

	//std::cout << "Size of unsigned char: " << sizeof(unsigned char) << '\n';
	//std::cout << "Size of short: " << sizeof(short) << '\n';

	//char* filename = (char*)"P:\\U\\Some2DGame\\Py\\info\\0.npy";
	//char* result = readFileBytes(filename);

	//unsigned char c1 = result[0];
	//unsigned char c2 = result[1];
	//printf("%x\n", c1);
	//printf("%x\n", c2);

	////printf("%x\n", result[0]);
	////std::cout << "1st symbol: " << (unsigned int)result[0] << '\n';
	//return 0;

	//char* filename = (char*)"P:\\U\\Some2DGame\\Py\\info\\0.npy";
	//char* result = readFileBytes(filename);

	//unsigned char firstUChar;
	//std::memcpy(&firstUChar, result, 1);

	//if (firstUChar != 147)
	//	return 1;

	//short lengthOfHeader;
	//std::memcpy(&lengthOfHeader, result + 8, 2);

	//char* finalArray = result + 10 + lengthOfHeader;

	//printf("1. %u\n", finalArray[0]);
	//printf("2. %u\n", finalArray[1]);
	//printf("3. %u\n", finalArray[2]);
	//printf("4. %u\n", finalArray[3]);

	////printf("%x\n", firstUChar);
	//std::cout << "firstUChar :" << firstUChar << '\n';
	//std::cout << "lengthOfHeader :" << lengthOfHeader << '\n';
	///*for (size_t i = 0; i < 16; i++)
	//{
	//	std::cout << "Byte #" << i << ": " << (int)(unsigned char)&result[i] << '\n';
	//}*/

	//delete []result;
}

char* readFileBytes(const char* name)
{
	std::ifstream fl(name);
	fl.seekg(0, std::ios::end);
	size_t len = fl.tellg();
	char* ret = new char[len];
	fl.seekg(0, std::ios::beg);
	fl.read(ret, len);
	fl.close();
	return ret;
}