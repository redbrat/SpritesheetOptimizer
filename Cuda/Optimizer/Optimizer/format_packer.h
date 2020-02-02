#pragma once
#include <tuple>
class format_packer
{
	public:
		static std::tuple<char*, int> pack(int atlasLength, char* atlasBuffer, char* offsetsBuffer);
};

