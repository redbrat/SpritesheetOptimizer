#include "file_reader.h"
#include <fstream>
#include <tuple>

std::tuple<char*, int> file_reader::readFile(std::string path)
{
	std::ifstream fl(path.c_str(), std::ios::binary);
	fl.seekg(0, fl.end);
	size_t len = fl.tellg();
	char* ret = new char[len];
	printf("ret len = %d\n", len);
	fl.seekg(0, fl.beg);
	fl.read(ret, len);
	fl.close();
	return { ret, len };
}
