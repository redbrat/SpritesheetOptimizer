#include "string_extensions.h"

char* string_extensions::makeSubset(const char* arr, int start, int end)
{
	if (end - start < 1)
		throw "end <= start. Can't make subset";
	char* to = (char*)malloc(end - start + 1);
	strncpy(to, arr + start, end - start);
	to[end - start] = '\0';
	return to;
}
