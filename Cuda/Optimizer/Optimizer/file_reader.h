#pragma once
#include <string>
#include <tuple>
class file_reader
{
	public:
		static std::tuple<char*, int> readFile(std::string path);
};

