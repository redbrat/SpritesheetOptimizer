#pragma once
#include <string>
#include <tuple>
#include <vector>
class numpy_reader
{
	public:
		static std::tuple<char*, int, std::vector<int>> readInt8(std::string path);
		static std::tuple<int*, int, std::vector<int>> readInt32(std::string path);
};

