#pragma once
#include<windows.h>
class FileGetter
{
	WIN32_FIND_DATAA found;
	HANDLE hfind;
	char folderstar[255];
	int chk;

	public:
		FileGetter(const char* folder);
		int getNextFile(char* fname);
};

