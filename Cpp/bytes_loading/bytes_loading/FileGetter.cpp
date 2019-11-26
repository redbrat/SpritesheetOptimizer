#include "FileGetter.h"

	FileGetter::FileGetter(const char* folder) {
		//sprintf(folderstar, "%s\\*.*", folder);
		hfind = FindFirstFileA(folderstar, &found);
		//skip .
		FindNextFileA(hfind, &found);
	}

	int FileGetter::getNextFile(char* fname) {
		//skips .. when called for the first time
		chk = FindNextFileA(hfind, &found);
		if (chk)
			strcpy(fname, found.cFileName);
		return chk;
	}