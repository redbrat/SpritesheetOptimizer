#include "bit_converter.h"

#include <fstream>

short bit_converter::GetShort(char* bytes, int offset)
{
	short result;
	std::memcpy(&result, bytes + offset, 2);
	return result;
}

int bit_converter::GetInt(char* bytes, int offset)
{
	int result;
	std::memcpy(&result, bytes + offset, 4);
	return result;
}

char bit_converter::GetChar(char* bytes, int offset)
{
	char result;
	std::memcpy(&result, bytes + offset, 1);
	return result;
}

unsigned char bit_converter::GetUnsignedChar(char* bytes, int offset)
{
	unsigned char result;
	std::memcpy(&result, bytes + offset, 1);
	return result;
}
