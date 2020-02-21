#pragma once
#include <tuple>
class format_packer
{
	public:
		static std::tuple<char*, int, unsigned int> pack(int atlasLength, char* atlasBuffer, unsigned int* offsetsBuffer, int spritesCount, short* spriteWidths, short* spriteHeights, char* prefixBuffer, int prefixLength);
};

