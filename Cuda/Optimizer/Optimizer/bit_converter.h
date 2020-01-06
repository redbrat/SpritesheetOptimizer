#pragma once
class bit_converter
{
public:
	static short GetShort(char* bytes, int offset);
	static unsigned short GetUnsignedShort(char* bytes, int offset);
	static int GetInt(char* bytes, int offset);
	static unsigned int GetUnsignedInt(char* bytes, int offset);
	static char GetChar(char* bytes, int offset);
	static unsigned char GetUnsignedChar(char* bytes, int offset);
};

